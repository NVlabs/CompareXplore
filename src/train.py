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
# Copyright (c) 2022, UCLA VAST Lab (GNN-DSE)
# Copyright (c) 2023, UCLA VAST Lab (HARP)
# BSD 3-Clause License (see LICENSE-THIRD-PARTY for full text)
#
# Portions provided under the following terms:
# Copyright (c) UCLA-DM (HLSyn)
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)
# (see LICENSE-THIRD-PARTY for full text)

from test import test
from pairwise import get_pairwise_data_loaders, PairwiseDataloader

from config import FLAGS
from saver import saver, NoOpContextManager
from utils import (
    OurTimer,
    _get_y_with_target,
    create_loss_dict_get_target_list,
    update_loss_dict,
    get_num_graphs,
    format_loss_dict,
)
from data import get_kernel_samples, split_dataset, torch_generator, get_num_designs

from model import feature_extract, check_feature_extract
from model_factory import create_model

import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join, basename
from accelerate import Accelerator
from transformers import get_scheduler
import math

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import bitsandbytes as bnb
from accelerate.utils import gather
from pprint import pformat


def train_main(dataset):
    # saver.info(f'Reading dataset from {SAVE_DIR}')

    # from torch.utils.data import random_split  # TODO: inductive

    if not FLAGS.all_kernels:
        dataset = get_kernel_samples(dataset)

    train_loaders = []
    if FLAGS.pairwise_class and 'pariwise_comparison' in FLAGS.pairwise_what_branches and 'dedicated' in FLAGS.pairwise_train_schemes:
        # if FLAGS.force_regen_pairwise_data:
        #     pair_dict = _gather_eval_pair_data(dataset)
        # else:
        #     pair_dict = None
        train_loader, val_loader, test_loader, _ = get_pairwise_data_loaders(
            dataset,
            torch_generator,
            pragma_differing_by=FLAGS.train_time_pragma_differing_by[0],
        )
        FLAGS.train_time_pragma_differing_by.pop(0)
        # for data in train_loader:
        #     print(data)
        # exit()

        if 'regression' in FLAGS.pairwise_what_branches and FLAGS.loss_components in [
            'both',
            'regression_only',
        ]:
            train_loader_regular, *_ = split_dataset(
                dataset
            )  # TODO: use which val_loader? val_loader.dataset only 1 batch????????????????
            train_loaders.append([train_loader_regular, 'r'])
        else:
            pass  # only having the comparison loss

        train_loaders.append([train_loader, 'p'])
    else:
        train_loader, val_loader, test_loader, _ = split_dataset(dataset)
        train_loaders.append([train_loader, 'r'])

    for train_loader, train_loader_type in train_loaders:
        saver.log_info(
            f'train loader {train_loader_type} has #designs={get_num_designs(train_loader)}'
        )

    model = create_model(dataset, train_loaders[-1][0])

    # if hasattr(FLAGS, 'OpenAI_embedding_model'):
    #     model.bert_model.save_cache()

    saver.log_model_architecture(model)
    saver.log_info(f'saver.log_model_architecture done')

    if FLAGS.sequence_modeling:
        if FLAGS.use_peft:
            model = do_peft(model)

    if hasattr(model, 'bert_model'):
        print_trainable_parameters(model.bert_model, 'model.bert_model')
    print_trainable_parameters(model, 'whole model')

    if FLAGS.sequence_modeling and not FLAGS.finetune_bert:
        for name, param in model.bert_model.named_parameters():
            param.requires_grad = False
            saver.log_info(f'No fine tune bert: Freezing param {name}')

    if FLAGS.load_model != None and FLAGS.load_model != 'None':
        saver.load_trained_model(FLAGS.load_model, model)

        if (
            FLAGS.pairwise_class
            and FLAGS.loss_components == 'both'
            and FLAGS.fix_encoder_classMLPs
        ):
            saver.log_info(f'fix_encoder_classMLPs=True:')
            except_params = set(
                [f'MLPs.{x[0]}' for x in list(model.MLPs.named_parameters())]
            )
            for name, param in model.named_parameters():
                if name not in except_params:
                    saver.log_info(f'\tFixing parameter: {name}')
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    saver.log_info(f'\tAllow training: {name}')
    else:
        saver.info(f'FLAGS.load_model is None')

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    if hasattr(FLAGS, 'load_guidance_emb') and FLAGS.load_guidance_emb:
        guidance_emb_dict = torch.load(FLAGS.guidance_emb_path)
        guidance_emb_dict['stencil'] = guidance_emb_dict['stencil_stencil2d']
    else:
        guidance_emb_dict = None
    # if not FLAGS.multi_target:
    #     model = Net(num_features).to(FLAGS.device)
    # else:
    #     model = NetMultiObjective(num_features).to(FLAGS.device)
    # print(model)

    optimizer = create_optimizer(model)
    saver.log_info(f'create_optimizer done')

    lr_scheduler = create_lr_scheduler(train_loader, optimizer)
    saver.log_info(f'get_scheduler done')

    # accelerator = Accelerator(gradient_accumulation_steps=FLAGS.gradient_accumulation_steps, split_batches=True)

    if FLAGS.mode == 'acc_launch':
        accelerator = saver.accelerator

        saver.log_info(
            f'accelerator creation done; accelerator.device={accelerator.device}'
        )
    else:
        saver.log_info(f'saver.accelerator={saver.accelerator}')

    # model.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    timer = OurTimer()
    if FLAGS.mode == 'acc_launch':
        model, optimizer, val_loader, test_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, val_loader, test_loader, lr_scheduler
        )

    if FLAGS.mode == "acc_launch":
        for i in range(len(train_loaders)):
            train_loader = train_loaders[i][0]
            # saver.log_info(
            #     f'{i}\'s train loader before accelerator.prepare; type(train_loader)={type(train_loader)}')
            train_loader = accelerator.prepare(train_loader)
            # saver.log_info(
            # f'{i}\'s train loader after accelerator.prepare; type(train_loader)={type(train_loader)}')
            train_loaders[i][0] = train_loader
        saver.log_info(f'accelerator.prepare done {timer.time_and_clear()}')

    # Initialize to optimal attention weights:
    # model.pool1.weight.data = torch.tensor([0., 1., 0., 0.]).view(1,4).to(device)
    train_losses = []
    val_losses = []
    test_losses = []
    gae_train_losses = []
    gae_val_losses = []
    gae_test_losses = []
    epochs = range(FLAGS.epoch_num)
    plot_test = False
    stagnant_epoch_num = 0
    weight_switch = False
    for epoch in epochs:

        # Achieve curriulum learning, replace the pairwise data loader with new ones.
        train_loaders = _it_is_time_to_update_pairwise_loaders(
            epoch, train_loaders, dataset
        )

        print(saver.logdir)
        print(basename(saver.logdir))
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Epoch {epoch} starts')

        # val
        if FLAGS.ignore_validation:
            saver.log_info('Ignore validation')
            val = 0.0
            loss_dict_val = {}
            gae_loss_val = 0.0
        else:
            if len(val_loader) > 0:
                saver.log_info(f'Epoch {epoch} val')
                val, loss_dict_val, gae_loss_val, _ = test(
                    val_loader,
                    'val',
                    model,
                    epoch,
                    forward_pairwise=FLAGS.pairwise_class, # tricky: should turn it on
                    eval_pairwise=False,
                )
                if hasattr(FLAGS, 'weight_switch') and FLAGS.weight_switch == True:
                    val = float(loss_dict_val['true_perf']) # deprecated

                if FLAGS.mode == 'standalone' or saver.accelerator.is_main_process:
                    saver.writer.add_scalar('val/val', val, epoch)

                if FLAGS.save_model and epoch % FLAGS.save_every_epoch == 0:
                    saver.save_trained_model(model, f"_{epoch}")

                all_val = gather(torch.tensor(val))
                val = all_val.mean()
                saver.log_info_at_most(
                    f'all_val={all_val}; val={val}', f'gather stuff', 1
                )
                val = val.item()

                if val_losses and val < min(val_losses):
                    if FLAGS.save_model:
                        saver.log_info((f'Saved val model at epoch {epoch}'))
                        # torch.save(model.state_dict(), join(saver.logdir, "val_model_state_dict.pth"))
                        saver.save_trained_model(
                            model,
                            path=join(saver.logdir, "val_model_state_dict.pth"),
                            info={'epoch': epoch},
                        )
                        # saver.save_trained_model(model, f"_val_{epoch}")
                    plot_test = True
                    stagnant_epoch_num = 0
                else:
                    stagnant_epoch_num += 1
            else:
                val = 0.0  # somehow val_loader is empty, e.g. DEBUG=True

        # test
        if FLAGS.ignore_testing:
            saver.log_info_once('Ignore testing')
            testr = 0.0
            loss_dict_test = {}
            gae_loss_test = 0.0
        else:
            raise NotImplementedError()  # deprecated below

        if (
            val_loader is not None
            and len(val_loader) > 0
            and test_loader is not None
            and len(test_loader) > 0
        ):
            saver.log_info((f'Val GAE loss: {gae_loss_val}'))
            saver.log_info((f'Val loss breakdown {format_loss_dict(loss_dict_val)}'))
            saver.log_info((f'Test GAE loss: {gae_loss_test}'))
            saver.log_info((f'Test loss breakdown {format_loss_dict(loss_dict_test)}'))
            val_losses.append(val)
            test_losses.append(testr)
            gae_val_losses.append(gae_loss_val)
            gae_test_losses.append(gae_loss_test)
        elif test_loader is not None and len(test_loader) > 0:
            saver.log_info((f'Test GAE loss: {gae_loss_test}'))
            saver.log_info((f'Test loss breakdown {format_loss_dict(loss_dict_test)}'))
            test_losses.append(testr)
            gae_test_losses.append(gae_loss_test)
        elif val_loader is not None and len(val_loader) > 0:
            saver.log_info((f'Val GAE loss: {gae_loss_val}'))
            saver.log_info((f'Val loss breakdown {format_loss_dict(loss_dict_val)}'))
            val_losses.append(val)
            gae_val_losses.append(gae_loss_val)

        # train
        saver.log_info(f'Epoch {epoch} train')
        train_loader_chosen, train_loader_name = train_loaders[
            epoch % len(train_loaders)
        ]
        forward_pairwise = FLAGS.pairwise_class
        if len(train_loaders) > 1 or FLAGS.pairwise_class:
            # saver.writer.add_text(
            #     'train/train_loader_chosen', train_loader_name, epoch)
            if train_loader_name == 'r':
                forward_pairwise = False
            saver.log_info(f'Epoch {epoch}: Choose {train_loader_name}; forward_pairwise={forward_pairwise}')

        loss, loss_dict_train, gae_loss_train, _ = train_one_epoch(
            epoch,
            model,
            train_loader_chosen,
            optimizer,
            lr_scheduler,
            forward_pairwise,
            weight_switch=weight_switch,
            guidance_emb_dict=guidance_emb_dict,
        )
        if hasattr(FLAGS, 'weight_switch') and FLAGS.weight_switch == True and loss < 1:
            weight_switch = True

        # To be efficient, let's not save train.
        # if train_losses and loss < min(train_losses):
        #     if FLAGS.save_model:
        #         saver.log_info((f'Saved train model at epoch {epoch}'))
        #         # torch.save(model.state_dict(), join(saver.logdir, "train_model_state_dict.pth"))
        #         saver.save_trained_model(model, path=join(saver.logdir, "train_model_state_dict.pth"),
        #                                  info={'epoch': epoch})
        # plot_test = True

        saver.log_info((f'Train GAE loss: {gae_loss_train}'))
        saver.log_info((f'Train loss breakdown {format_loss_dict(loss_dict_train)}'))

        # if len(val_loader) > 0 and len(test_loader) > 0:
        saver.log_info(
            (
                'Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, '
                'Test: {:.4f}) Time: {}'.format(
                    epoch, loss, val, testr, timer.time_and_clear()
                )
            )
        )

        train_losses.append(loss)
        if FLAGS.mode == 'standalone' or saver.accelerator.is_main_process:
            saver.writer.add_scalar('loss/loss_epoch', loss, epoch)

            for loss_name, loss_value in loss_dict_train.items():
                saver.writer.add_scalar(
                    f'loss/loss_breakdown_{loss_name}', loss_value, epoch
                )

            saver.writer.add_scalar(f'loss/loss_epoch_{train_loader_name}', loss, epoch)

        gae_train_losses.append(gae_loss_train)

        if len(train_losses) > 50:
            if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                break
        if (
            val_losses
            and hasattr(FLAGS, "max_stagnant_epochs")
            and stagnant_epoch_num >= FLAGS.max_stagnant_epochs
            and epoch > FLAGS.epoch_num // 2
        ):
            # early stopping
            saver.log_info(
                f'Early stopping! stagnant_epoch_num={stagnant_epoch_num}; FLAGS.max_stagnant_epochs={FLAGS.max_stagnant_epochs}; epoch={epoch}'
            )
            break

    if FLAGS.epoch_num == 0:
        return

    epochs = range(epoch + 1)
    import matplotlib

    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

    plt.plot(range(len(train_losses)), train_losses, 'g', label='Training loss')
    if val_loader is not None and len(val_loader) > 0:
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
    if test_loader is not None and len(test_loader) > 0:
        plt.plot(epochs, test_losses, 'r', label='Testing loss')
    plt.title('Training, Validation, and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(saver.get_log_dir(), 'losses.png'), bbox_inches='tight')
    plt.close()

    if FLAGS.gae_T or FLAGS.gae_P:
        plt.plot(epochs, gae_train_losses, 'g', label='Training loss')
        if val_loader is not None and len(val_loader) > 0:
            plt.plot(epochs, gae_val_losses, 'b', label='Validation loss')
        if test_loader is not None and len(test_loader) > 0:
            plt.plot(epochs, gae_test_losses, 'r', label='Testing loss')
        plt.title('Training, Validation, and Testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(join(saver.get_log_dir(), 'gae_losses.png'), bbox_inches='tight')
        plt.close()
    if test_loader is not None and len(test_loader) > 0:
        saver.log_info(f'min test loss at epoch: {test_losses.index(min(test_losses))}')
    if val_loader is not None and len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses))}')
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')

    if hasattr(FLAGS, 'OpenAI_embedding_model'):
        model.bert_model.save_cache()


def _it_is_time_to_update_pairwise_loaders(epoch, train_loaders, dataset):
    if FLAGS.pairwise_class and 'pariwise_comparison' in FLAGS.pairwise_what_branches and 'dedicated' in FLAGS.pairwise_train_schemes:

        train_loader_replaced = False
        if len(FLAGS.train_time_pragma_differing_by) > 0:
            conf = FLAGS.train_time_pragma_differing_by[0]

            assert type(conf) is tuple and len(conf) == 3
            if epoch > 0 and epoch >= conf[1]:
                saver.log_info(
                    f'pairwise conf={conf}; current epoch={epoch}; epoch >= conf[1], so it is time to replace the pairwise train data loader (part of the curriculum learning scheme)'
                )
                train_loader_pairwise_new, _, _, _ = get_pairwise_data_loaders(
                    dataset, torch_generator, pragma_differing_by=conf
                )
                if FLAGS.mode == "acc_launch":
                    train_loader_pairwise_new = saver.accelerator.prepare(
                        train_loader_pairwise_new
                    )

                replaced = False
                for i in range(len(train_loaders)):
                    if train_loaders[i][1] == 'p':
                        assert not replaced
                        train_loaders[i] = [train_loader_pairwise_new, 'p']
                        replaced = True
                        break
                assert replaced
                saver.log_info(f'epoch={epoch}; replaced pairwise train loader')
                FLAGS.train_time_pragma_differing_by.pop(0)

                train_loader_replaced = True


        if not train_loader_replaced:
            if epoch > 0:
                # Always replace it -- reason is very tricky to explain -- due to torch 
                # data loader caching the underlying dataset
                # and we want to subsample at each new eppch
                # If we use the full list, each epoch will take too long.

                # First, find the current pairwise data loader
                train_loader_pairwise_cur = None
                for i in range(len(train_loaders)):
                    if train_loaders[i][1] == 'p':
                        train_loader_pairwise_cur = train_loaders[i][0]
                        break
                assert train_loader_pairwise_cur is not None

                train_loader_pairwise_new = PairwiseDataloader(
                    existing_dataloader=train_loader_pairwise_cur
                )
                train_loaders[i] = (train_loader_pairwise_new, 'p') # replaced

    return train_loaders


def create_optimizer(model):
    optimizer = _diff_lrs_for_diff_branches_HARP(model)
    if optimizer is not None:
        return optimizer

    if FLAGS.opt_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
        )
    elif FLAGS.opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
        )
    else:
        raise ValueError(f'FLAGS.opt_type {FLAGS.opt_type} unrecognized')
    return optimizer


def _diff_lrs_for_diff_branches_HARP(model):
    if FLAGS.load_model_HARPnet and FLAGS.subtask == 'train' and FLAGS.eval_pairwise and 'pariwise_comparison' in FLAGS.pairwise_what_branches and FLAGS.HARP_different_lrs:

        param_groups = []
        # Define the parameter groups and their corresponding learning rates
        base_params = []
        pairwise_class_params = []
        
        for name, param in model.named_parameters():
            if 'MLPs_pairwise_class' in name:
                pairwise_class_params.append((name, param))
            else:
                base_params.append((name, param))
        
        param_groups.append({'params': [p for _, p in base_params], 'lr': FLAGS.base_lr})
        param_groups.append({'params': [p for _, p in pairwise_class_params], 'lr': FLAGS.pairwise_class_lr})
        
        # Create a summary of param_groups with parameter names
        summary = [
            {
                'group_name': 'base_params' if i == 0 else 'pairwise_class_params',
                'lr': group['lr'],
                'num_params': len(group['params']),
                'param_names': [name for name, _ in (base_params if i == 0 else pairwise_class_params)]
            }
            for i, group in enumerate(param_groups)
        ]
        
        saver.log_info(f"Parameter Groups Summary:\n{pformat(summary)}")
        
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer


    return None


def create_lr_scheduler(train_loader, optimizer):
    if FLAGS.max_train_steps is None:
        # See https://github.com/huggingface/diffusers/issues/6137
        accelerator_num_processes = 1

        num_update_steps_per_epoch = math.ceil(
            len(train_loader)
            / accelerator_num_processes
            / FLAGS.gradient_accumulation_steps
        )

        saver.log_info(
            f'create_lr_scheduler: num_update_steps_per_epoch = math.ceil(en(train_loader) / accelerator_num_processes / FLAGS.gradient_accumulation_steps); {num_update_steps_per_epoch} = math.ceil({len(train_loader)} / {accelerator_num_processes} / {FLAGS.gradient_accumulation_steps})'
        )

        max_train_steps = FLAGS.epoch_num * num_update_steps_per_epoch

        saver.log_info(
            f'create_lr_scheduler: max_train_steps = FLAGS.epoch_num * num_update_steps_per_epoch; {max_train_steps} = {FLAGS.epoch_num} * {num_update_steps_per_epoch}'
        )
    else:
        max_train_steps = FLAGS.max_train_steps

    num_warmup_steps = FLAGS.num_warmup_steps * FLAGS.gradient_accumulation_steps

    saver.log_info(
        f'create_lr_scheduler: num_warmup_steps=FLAGS.num_warmup_steps * FLAGS.gradient_accumulation_steps; {num_warmup_steps} = {FLAGS.num_warmup_steps} * {FLAGS.gradient_accumulation_steps}'
    )

    num_training_steps = max_train_steps * FLAGS.gradient_accumulation_steps

    saver.log_info(
        f'create_lr_scheduler: num_training_steps=max_train_steps * FLAGS.gradient_accumulation_steps; {num_training_steps} = {max_train_steps} * {FLAGS.gradient_accumulation_steps}'
    )

    lr_scheduler = get_scheduler(
        name=FLAGS.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return lr_scheduler


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    lr_scheduler,
    forward_pairwise,
    task=FLAGS.task,
    weight_switch=False,
    guidance_emb_dict=None,
):
    model.train()

    total_loss = 0
    correct = 0
    i = 0
    num_data = 0

    loss_dict, target_list = create_loss_dict_get_target_list(FLAGS, task)

    # if type(train_loader) is PairwiseDataloader:
    #     # Update the pairwise train loader.
    #     train_loader.update_generator()

    for iter_id, data in enumerate(tqdm(train_loader)):
        if FLAGS.mode == 'acc_launch':
            context = saver.accelerator.accumulate(model)
        else:
            context = NoOpContextManager()

        with context:
            # model.to(accelerator.device)

            num_data += get_num_graphs(data)
            if FLAGS.debug_iter != -1 and iter_id > FLAGS.debug_iter:
                saver.log_info(
                    f'Debugging mode: iter_id={iter_id} > FLAGS.debug_iter={FLAGS.debug_iter}; stop the epoch'
                )
                break
        # data = data.to(FLAGS.device)
        optimizer.zero_grad()
        data.weight_switch = weight_switch

        optimizer.zero_grad()

        if FLAGS.mode == 'acc_launch':
            pass
        else:
            assert FLAGS.mode == 'standalone'
            data = data.to(FLAGS.device)


        saver.log_info_at_most(f'train_one_epoch: check gname if mixed; data.gname={data.gname}; len(train_loader)={len(train_loader)}', 'toecgim', 1)

        if forward_pairwise:
            saver.log_info_at_most(f'train_one_epoch: check unqiue_id if subsampling is correct; data.unique_id[0:10]={data.unique_id[0:10]}', 'toe-uid', 10)


        out, loss, loss_dict_, gae_loss = model(
            data, forward_pairwise=forward_pairwise, tvt='train'
        )

        if guidance_emb_dict is not None:
            assert FLAGS.load_pretrained_GNN == False
            gnames = data.gname
            guide_node_emb = [
                guidance_emb_dict[gname].to(FLAGS.device) for gname in gnames
            ]
            guide_node_emb = torch.cat(guide_node_emb, dim=0)

            node_emb = out['node_emb']
            x = data.x_programl
            mask = (x[:, 0] + x[:, 1]) > 0
            node_emb = node_emb[mask]
            assert (
                node_emb.shape == guide_node_emb.shape
            ), f'node_emb.shape={node_emb.shape}; guide_node_emb.shape={guide_node_emb.shape}'
            guide_loss = torch.mean(
                1 - F.cosine_similarity(node_emb, guide_node_emb.detach())
            )
            # print(guide_loss)
            loss += FLAGS.guide_loss_w * guide_loss
            loss_dict_['guide_loss'] = guide_loss

        if FLAGS.mode == 'acc_launch':
            saver.accelerator.backward(loss)
        else:
            assert loss > 0
            loss.backward()

        if (
            hasattr(FLAGS, 'max_grad_norm')
            and FLAGS.max_grad_norm is not None
            and FLAGS.max_grad_norm > 0
        ):
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_grad_norm)
            saver.log_info_at_most(
                f'torch.nn.utils.clip_grad_norm_ FLAGS.max_grad_norm={FLAGS.max_grad_norm}',
                msg_type='gclip',
                times=1,
            )

        if FLAGS.task == 'regression':
            total_loss += loss.item()# * get_num_graphs(data)
            if not FLAGS.SSL:
                loss_dict = update_loss_dict(
                    loss_dict, loss_dict_, target_list, FLAGS, data
                )
        else:
            loss_, pred = torch.max(out[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total_loss += labels.size(0)
        optimizer.step()
        lr_scheduler.step()

    if FLAGS.task == 'regression':
        return (
            total_loss / len(train_loader),
            {key: v / len(train_loader) for key, v in loss_dict.items()},
            gae_loss.item(),
            num_data,
        )
    else:
        return (
            1 - correct / total_loss,
            {key: v / len(train_loader) for key, v in loss_dict.items()},
            gae_loss.item(),
            num_data,
        )


def do_peft(model):

    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning

    if 'codellama' in FLAGS.code_encoder:
        model.bert_model.gradient_checkpointing_enable()
        saver.log_info(f'gradient_checkpointing_enable()')

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    saver.log_info(
        f'is_loaded_in_8bit={getattr(model.bert_model, "is_loaded_in_8bit", False)}'
    )
    saver.log_info(
        f'is_loaded_in_4bit={getattr(model.bert_model, "is_loaded_in_4bit", False)}'
    )

    model.bert_model = prepare_model_for_kbit_training(model.bert_model)

    # model.bert_model = prepare_model_for_int4_training(model.bert_model)

    # Get lora module names
    # modules = find_all_linear_names(model.bert_model)
    # saver.log_info(f'find_all_linear_names: {modules}')

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config()
    model.bert_model = get_peft_model(model.bert_model, peft_config)

    # Print information about the percentage of trainable parameters

    return model


def create_peft_config():
    config = LoraConfig(
        r=FLAGS.peft_r,  # dimension of the updated matrices
        lora_alpha=FLAGS.lora_alpha,  # parameter for scaling
        target_modules='all-linear',
        lora_dropout=FLAGS.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def print_trainable_parameters(model, label, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    saver.log_info(
        f"{label}: all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():

        saver.log_info(f'name={name}; type(module)={type(module)}')

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if len(lora_module_names) == 0:
        raise RuntimeError(f'No modules in lora_module_names! {lora_module_names}')
    return list(lora_module_names)
