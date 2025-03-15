# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions provided under the following terms:
# Copyright (c) UCLA-DM
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)
# (see LICENSE-THIRD-PARTY for full text)


from train import train_one_epoch, create_optimizer, create_lr_scheduler
from test import test
from config import FLAGS
from saver import saver
from data import split_li, create_dataloader, shuffle_split_li_into_2_chunks

from model import feature_extract
from model_factory import create_model
from utils import deduce_load_model_path

import torch
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import random
import numpy as np
import copy
from transformers import get_scheduler


PLUSMINUS_SYMBOL = u"\u00B1"


def adapt_main(dataset, task):
    # The key characteristic of adaptation is that it is done per kernel,
    # i.e. for each new/unseen test kernel (inductive), adaptation is performed on it
    # by copying the model's weights and doing some adaptation.
    if FLAGS.tvt_split_by != 'kernels_inductive':
        saver.log_info(
            f'adapt_main: FLAGS.tvt_split_by != "kernels_inductive ({FLAGS.tvt_split_by})"')
        return

    dataloaders = _split_into_indiviual_dataloaders(dataset)

    eval_dict = OrderedDict()
    eval_dict['before_adaptation'] = OrderedDict()
    eval_dict['after_adaptation'] = OrderedDict()
    if 'dse' in FLAGS.subtask:
        adapted_models_dict = OrderedDict()
    else:
        adapted_models_dict = None

    if hasattr(FLAGS, 'load_guidance_emb') and FLAGS.load_guidance_emb:
        guidance_emb_dict = torch.load(FLAGS.guidance_emb_path)
        guidance_emb_dict['stencil'] = guidance_emb_dict['stencil_stencil2d']
    else:
        guidance_emb_dict = None

    for gname, dls in dataloaders.items():
        for k in eval_dict.keys():
            eval_dict[k][gname] = []
        if adapted_models_dict is not None:
            adapted_models_dict[gname] = []
        for j, dl in enumerate(dls):
            eval_dict, adapted_models_dict = _adapt_and_test(
                dl, f'{gname}_{j}', eval_dict, adapted_models_dict, gname, task, dataset, guidance_emb_dict = guidance_emb_dict)

    summary_str = f'{task}:'
    for key, eval_d in eval_dict.items():
        summary_str = saver.log_info(f'-' * 20, build_str=summary_str)
        final_print_dict = OrderedDict()
        summary_str = saver.log_info(f'{key}:', build_str=summary_str)
        all_rmses = []
        for gname, rmse_list in eval_d.items():
            rmse_list_s = [f'{x:.4f}' for x in rmse_list]
            fps = _avg_plusminus_std_for_li(rmse_list)
            summary_str = saver.log_info(f'\t{gname}: {rmse_list_s} --> {fps}', build_str=summary_str)
            final_print_dict[gname] = fps
            all_rmses += rmse_list
        all_agg_key = f'all/aggregated({len(all_rmses)})'
        keys_to_use = [all_agg_key] + list(final_print_dict.keys())
        final_print_dict[all_agg_key] = _avg_plusminus_std_for_li(all_rmses)
        ks = "\t".join(keys_to_use)
        vs = "\t".join([final_print_dict[k] for k in keys_to_use])
        summary_str = saver.log_info(f'Summary of adaptation for {key}:', build_str=summary_str)
        summary_str = saver.log_info(f'{ks}', build_str=summary_str)
        summary_str = saver.log_info(f'{vs}', build_str=summary_str)
        summary_str = saver.log_info(f'-' * 20, build_str=summary_str)
    saver.log_info(f'Adaptation done')
    # exit()
    adapt_result_dict = {'eval_dict': eval_dict, 'adapted_models_dict': adapted_models_dict}
    saver.save_dict(eval_dict, f'eval_dict_{task}.pkl', subfolder='obj')
    saver.log_info_new_file(summary_str, 'final_report.txt')
    return adapt_result_dict, summary_str


def _avg_plusminus_std_for_li(li):
    avg = np.mean(li)
    std = np.std(li)
    return f'{avg:.4f}{PLUSMINUS_SYMBOL}{std:.2f}'


def _adapt_and_test(dl, test_name, eval_dict, adapted_models_dict, gname, task, dataset, guidance_emb_dict = None):
    load_model = deduce_load_model_path(task, FLAGS)
    model = create_model(dl.adaptation_dl.dataset, None, task, load_model=load_model)
    # model = saver.accelerator.prepare(
    #     model
    # )

    if FLAGS.sequence_modeling and not FLAGS.finetune_bert:
        for name, param in model.bert_model.named_parameters():
            param.requires_grad = False
            saver.log_info(f'No fine tune bert: Freezing param {name}')


    if load_model != None and load_model != 'None':
        model, _ = saver.load_trained_model(load_model, model)

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    # otherwise, no need to test the model again and again
    if len(eval_dict['before_adaptation'][gname]) == 0:
        r_before_adapt = _test(model, dl, f'{test_name}_before', task, dataset=dataset)
        eval_dict['before_adaptation'][gname].append(r_before_adapt)

    if FLAGS.num_mini_epochs >= 1:
        model = _adapt(model, dl, task)
        r_after_adapt = _test(model, dl, f'{test_name}_after', task, dataset=dataset)
        eval_dict['after_adaptation'][gname].append(r_after_adapt)
        if adapted_models_dict is not None:
            adapted_models_dict[gname].append(model)
        
    # saver.log_info(f'adapted_models_dict={adapted_models_dict}')

    return eval_dict, adapted_models_dict


def _test(model, dl, test_name, task, dataset):
    testr, loss_dict, encode_loss, eval_df_dict = test(dl.final_hold_out_dl, 'test', model, epoch=0, plot_test=True,
                                                  csv_dict=None,
                                                  data_dict=None, forward_pairwise=False,
                                                  eval_pairwise=False, test_name=f'{test_name}_holdout', task=task, dataset=dataset)

    # saver.log_info(f'testr={testr}')
    # saver.log_info(f'loss_dict={loss_dict}')
    # saver.log_info(f'encode_loss={encode_loss}')
    # print('@@@', eval_df)
    # exit(-1)
    if task == 'regression':
        assert eval_df_dict['point'].iloc[-1]['target'] == 'tot/avg'
        rmse = eval_df_dict['point'].iloc[-1]['rmse']
        rtn = rmse
    elif task == 'class':
        accuracy = eval_df_dict['point']['accuracy']
        rtn = accuracy
    else:
        raise NotImplementedError()

    return rtn


def _adapt(model, dl, task, guidance_emb_dict = None):
    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(dl.adaptation_dl, optimizer)

    assert FLAGS.num_mini_epochs >= 1
    train_losses = []
    val_losses = []
    model_best = None
    weight_switch = False

    for epoch in range(FLAGS.num_mini_epochs):
        loss, loss_dict_train, gae_loss_train, num_data = train_one_epoch(
            epoch, model, dl.adaptation_dl, optimizer, lr_scheduler, forward_pairwise=False, task=task, weight_switch = weight_switch, guidance_emb_dict = guidance_emb_dict)
        # loss, loss_dict_train, gae_loss_train, num_data = train(epoch, model, dl.adaptation_dl, optimizer,
        #                                                         forward_pairwise=False, weight_switch = weight_switch, guidance_emb_dict = guidance_emb_dict)
        # val, loss_dict, encode_loss, eval_df = test(dl.adaptation_valid_dl, 'valid', model, epoch=0, plot_test=True,
                                                #   csv_dict=None,
                                                #   data_dict=None, forward_pairwise=False,
                                                #   eval_pairwise=False)


        if hasattr(FLAGS, 'weight_switch') and FLAGS.weight_switch == True and loss < 1:
            weight_switch = True

        if FLAGS.test_which_adapted_model == 'best_train' and (model_best is None or (train_losses and loss < min(train_losses))):
            # print('FLAGS.test_which_adapted_model',
            #       FLAGS.test_which_adapted_model)
            model_best = copy.deepcopy(model)
        if FLAGS.test_which_adapted_model == 'best_valid' and (model_best is None or (val_losses and val < min(val_losses))):
            print('FLAGS.test_which_adapted_model', FLAGS.test_which_adapted_model)
            model_best = copy.deepcopy(model)

        train_losses.append(loss)
        saver.log_info(
            f'Mini epoch {epoch}: loss={loss:.4f} num_data={num_data}')
        saver.log_info(
        f'min train loss at epoch: {train_losses.index(min(train_losses))}')
        # val_losses.append(val)

        # saver.log_info(f'Mini epoch {epoch}: loss={loss:.4f} num_data={num_data}, valid loss {val:.4f}')
        saver.log_info(f'Mini epoch {epoch}: loss={loss:.4f} num_data={num_data}')

    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')

    if FLAGS.test_which_adapted_model == 'last_epoch':
        pass
    elif FLAGS.test_which_adapted_model in ['best_train', 'best_valid']:
        assert model_best is not None
        model = model_best  # return the best model in training
        saver.log_info(f'Returning the best model according to train loss')
    else:
        raise NotImplementedError()
    return model


class AdaptationDataLoaders(object):
    def __init__(self, final_hold_out_dl, adaptation_dl):
        self.final_hold_out_dl = final_hold_out_dl
        self.adaptation_dl = adaptation_dl
        # self.adaptation_valid_dl = adaptation_valid_dl


def _split_into_indiviual_dataloaders(dataset):
    saver.log_info(
        f'Split the dataset by invidual kernels according to {FLAGS.test_kernels}')
    # train_file_li = []
    test_kernel_li_map = defaultdict(list)
    file_li = dataset.processed_file_names
    for file in tqdm(file_li):
        data = torch.load(file)
        if data.gname in FLAGS.test_kernels:
            test_kernel_li_map[data.gname].append(file)
            # train_file_li.append(file)
    saver.log_info(f'-' * 20)

    # saver.log_info(f'Found {len(train_file_li)} files for training')

    dataloaders = OrderedDict()
    for gname, test_file_li in sorted(test_kernel_li_map.items()):
        if len(test_file_li) <= 2:
            raise RuntimeError(f'Must have at least 2 points! Instead len(test_file_li)={len(test_file_li)}; test_file_li={test_file_li}')

        li = shuffle_split_li_into_2_chunks(
            test_file_li, ratio=FLAGS.test_holdout_ratio)
        saver.log_info(
            f'\tval/test: \t{gname} has {len(test_file_li)} designs in total; split into others {len(li[1])} and final test {len(li[0])}')
        final_hold_out_data_points = li[0]
        if len(final_hold_out_data_points) == 0:
            raise RuntimeError(f'len(final_hold_out_data_points) == 0')
        final_hold_out_dl = create_dataloader(
            final_hold_out_data_points, shuffle=False)

        assert FLAGS.repeat_times >= 1
        rest_for_choosing = li[1]
        if FLAGS.adaptation_num_dp == 'all':
            adaptation_num_dp = len(rest_for_choosing)
        else:
            adaptation_num_dp = FLAGS.adaptation_num_dp
        saver.log_info(f'Sampling {adaptation_num_dp} out of {len(rest_for_choosing)} for {FLAGS.repeat_times} '
                       f'times to get adaptation data points')
        if len(rest_for_choosing) < adaptation_num_dp:
            saver.log_info_once(f'Not enough rest designs to choose {len(rest_for_choosing)} < {FLAGS.adaptation_num_dp}')
            adaptation_num_dp = len(rest_for_choosing)
            
        dataloaders[gname] = []
        for i in range(FLAGS.repeat_times):
            saver.log_info(f'\tRepeat {i}: creating dataloaders')

            if len(rest_for_choosing) <= 1:
                raise RuntimeError('Must have at least 1 data point!')
            li = shuffle_split_li_into_2_chunks(
                rest_for_choosing, num=adaptation_num_dp, force_shuffle=True)
            adaptation_data_points = li[0]
            assert len(adaptation_data_points) == adaptation_num_dp
            adaptation_dl = create_dataloader(
                adaptation_data_points, shuffle=FLAGS.shuffle)

            dataloaders[gname].append(AdaptationDataLoaders(
                final_hold_out_dl, adaptation_dl))

    return dataloaders
