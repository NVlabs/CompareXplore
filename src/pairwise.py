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

from data import SAVE_DIR, MyOwnDataset, split_dataset, shuffle_split_li_into_2_chunks
from data_src_code import read_source_code, get_text_representation_of_design, tokenizer
from config import FLAGS
from saver import saver
from utils import (
    create_dir_if_not_exists,
    create_pred_dict,
    print_stats,
    report_save_dir,
    save_pickle,
    load_pickle,
)

from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

import math
import random
from tqdm import tqdm
import os
from os.path import join
import torch
from shutil import rmtree
from glob import glob
from collections import defaultdict, OrderedDict
import pandas as pd
from scipy.spatial import distance
from random import Random


def get_pairwise_data_loaders(
    dataset, torch_generator, data_dict=None, pragma_differing_by=1
):
    saver.log_info(
        f'get_pairwise_data_loaders: pragma_differing_by={pragma_differing_by}'
    )
    # assert not FLAGS.sequence_modeling
    # if FLAGS.force_regen_pairwise_data:
    # if FLAGS.split_pairs_by_holding_out == 'pairs':
    #     all_pair_li = []
    #     for gname, pair_li in pair_dict.items():
    #         all_pair_li += pair_li
    #
    #     train_li, val_li, test_li = _shuffle_a_list_split_into_3_tvt_chunks(all_pair_li, 'pairs')

    # elif FLAGS.split_pairs_by_holding_out == 'designs':

    train_ds, val_ds, test_ds, transductive_test_ds = split_dataset(
        dataset, batch_size=1, concise=False
    )  # to save memory, use bs=1
    train_design_li, val_design_li, test_design_li, transductive_test_design_li = (
        set(),
        set(),
        set(),
        set(),
    )

    
    for dl, tvt, design_li in tqdm(
        [
            (train_ds, 'train', train_design_li),
            (val_ds, 'val', val_design_li),
            (test_ds, 'test', test_design_li),
            (
                transductive_test_ds,
                'transductive_test',
                transductive_test_design_li,
            ),
        ]
    ):
        if tvt not in ['test', 'transductive_test'] and FLAGS.subtask != 'train':
            continue
        for d in dl:  # may OOM here

            # saver.log_info_once(f'pairwise d.x.device={d.x.device}')
            gnames = d.gname
            points = d.xy_dict_programl['point']

            for x, y in zip(gnames, points):
                design_li.add((x, y))
            # design_li += d.xy_dict_programl['point']
        saver.log_info(f'{tvt} design_li: {len(design_li)}')
        # assert len(design_li) != 0

    if FLAGS.subtask == 'train':
        _check_overlapping(
            train_design_li, val_design_li, 'train_design_li', 'val_design_li'
        )
        _check_overlapping(
            train_design_li, test_design_li, 'train_design_li', 'test_design_li'
        )
        _check_overlapping(val_design_li, test_design_li, 'val_design_li', 'test_design_li')
        _check_overlapping(
            transductive_test_design_li,
            test_design_li,
            'transductive_test_design_li',
            'test_design_li',
        )

    if data_dict is None:
        data_dict = get_data_dict_by_gname(dataset)
    pair_dict, *_ = gather_eval_pair_data(
        data_dict, pragma_differing_by=pragma_differing_by
    )  # also save csv to disk

    num_pairs_in_total = 0
    for gname, pair_li in pair_dict.items():
        for pair in pair_li:
            data1, data2, _, _ = pair
            d1_str = data1.xy_dict_programl['point']
            d2_str = data2.xy_dict_programl['point']
            assert d1_str != d2_str
            num_pairs_in_total += 1
    if num_pairs_in_total == 0:
        raise ValueError(f'num_pairs_in_total=0')

    # train_design_li, val_design_li, test_design_li = _shuffle_a_list_split_into_3_tvt_chunks(
    #     all_designs, 'individual designs')

    train_li, val_li, test_li, transductive_test_li = [], [], [], []
    for gname, pair_li in pair_dict.items():
        for pair in pair_li:
            data1, data2, _, _ = pair
            d1_str = data1.xy_dict_programl['point']
            d2_str = data2.xy_dict_programl['point']
            assert d1_str != d2_str
            if FLAGS.subtask == 'train':
                if (gname, d1_str) in train_design_li and (
                    gname,
                    d2_str,
                ) in train_design_li:
                    train_li.append(pair)
                if (gname, d1_str) in val_design_li and (
                    gname,
                    d2_str,
                ) in val_design_li:
                    val_li.append(pair)
            if (gname, d1_str) in test_design_li and (
                gname,
                d2_str,
            ) in test_design_li:
                test_li.append(pair)
            if (gname, d1_str) in transductive_test_design_li and (
                gname,
                d2_str,
            ) in transductive_test_design_li:
                transductive_test_li.append(pair)
    for li, li_name in [
        (train_li, 'train_design_li'),
        (val_li, 'val_design_li'),
        (test_li, 'test_design_li'),
        (transductive_test_li, 'transductive_test_design_li'),
    ]:
        saver.log_info(
            f'Pairs (both data1 and data2) that are in {li_name}: {len(li)} (/{num_pairs_in_total}={len(li) / num_pairs_in_total:.4%})'
        )

        if FLAGS.write_pairs_to_disk:
            _save_pair_li_as_csv(
                f'diff_{pragma_differing_by}_{li_name}', li, target_list=None
            )

    # else:
    #     assert False

    # if FLAGS.subtask == 'inference':  # to save time
    #     train_data_li = None
    #     val_data_li = None
    #     saver.log_info(f'Skipping train_data_li and val_data_li to be quick')
    # else:
    #     train_data_li = list(_get_pairwise_data_gen(train_li, 'train'))
    #     val_data_li = list(_get_pairwise_data_gen(val_li, 'val'))
    # test_data_li = list(_get_pairwise_data_gen(test_li, 'test'))

    # transductive_test_data_li = list(_get_pairwise_data_gen(transductive_test_li, 'transductive_test'))

    # else:
    #     # train_data_li, val_data_li, test_data_li, transductive_test_data_li = None, None, None, None
    #     pass

    if FLAGS.subtask == 'inference':  # to save time
        train_loader = None
        val_loader = None
        saver.log_info(f'Skipping train_loader and val_loader to be quick')
    else:
        train_loader = get_pairwise_data_loader(
            train_li, 'train', torch_generator, pragma_differing_by
        )
        val_loader = get_pairwise_data_loader(
            val_li, 'val', torch_generator, pragma_differing_by
        )

    if FLAGS.subtask == 'train':
        test_loader = None
        transductive_test_loader = None
    else:
        test_loader = get_pairwise_data_loader(
            test_li, 'test', torch_generator, pragma_differing_by
        )
        transductive_test_loader = get_pairwise_data_loader(
            transductive_test_li,
            'transductive_test',
            torch_generator,
            pragma_differing_by,
        )

    return train_loader, val_loader, test_loader, transductive_test_loader


def _check_overlapping(a, b, label1, label2):
    num_dup = len(set(a) & set(b))
    saver.log_info(
        f'{label1} ({len(a)}) and {label2} ({len(b)}) have {num_dup} overlapping/intersection'
    )



def get_pairwise_data_loader(data_li, tvt, torch_generator, pragma_differing_by):
    return PairwiseDataloader(
        pin_memory=FLAGS.pin_memory,
        generator=torch_generator,
        collate_fn=pairwise_custom_collate,
        data_li=data_li,
        pragma_differing_by=pragma_differing_by,
        tvt=tvt,
    )


class PairwiseDataloader(DataLoader):
    def __init__(
        self, existing_dataloader=None, pin_memory=None, generator=None, collate_fn=None, data_li=None, pragma_differing_by=None, tvt=None
    ):
        if existing_dataloader is not None:
            self.data_li = existing_dataloader.data_li
            self.pragma_differing_by = existing_dataloader.pragma_differing_by
            self.tvt = existing_dataloader.tvt
            self.epoch_cnt = existing_dataloader.epoch_cnt + 1
            self.num_pairs_at_most, self.num_pairs_in_total, self.correct_num_data = (
                _deduce_num_data(self.pragma_differing_by, self.data_li)
            )
            self.dataset = PairwiseDataset(None, self.correct_num_data)
            self.update_generator()
            super(PairwiseDataloader, self).__init__(
                dataset=self.dataset,
                pin_memory=existing_dataloader.pin_memory,
                generator=torch.Generator().manual_seed(FLAGS.random_seed + self.epoch_cnt),
                collate_fn=existing_dataloader.collate_fn,
            )
        else:

            self.data_li = data_li
            if tvt == 'train' and FLAGS.shuffle: # if test etc. no need to shuffle (might incur other tricky issues or make debugging look complicated)
                saver.log_info(f'PairwiseDataloader: shuffle data_li of {len(self.data_li)}')
                Random(FLAGS.random_seed).shuffle(self.data_li)
                gnames = [(pair[0].gname, pair[1].gname) for pair in self.data_li]
                saver.log_info(f'gnames[0:10]={gnames[0:10]}')


            self.pragma_differing_by = pragma_differing_by
            self.tvt = tvt
            self.epoch_cnt = 0
            self.num_pairs_at_most, self.num_pairs_in_total, self.correct_num_data = (
                _deduce_num_data(self.pragma_differing_by, self.data_li)
            )

            self.dataset = PairwiseDataset(
                None,
                self.correct_num_data,
            )
            self.update_generator()


            # self.dataset = PairwiseDataset(
            #     _get_pairwise_data_batch_yield(self._sample_data_li(), tvt),
            #     self.correct_num_data,
            # )

            super(PairwiseDataloader, self).__init__(
                dataset=self.dataset,
                pin_memory=pin_memory,
                generator=generator,
                collate_fn=collate_fn,
            )

            # self.num_designs = dataset.get_num_designs()

    def update_generator(self):
        new_sample = self._sample_data_li()
        self.dataset.generator = _get_pairwise_data_batch_yield(new_sample, self.tvt)

    

    def _sample_data_li(self):

        if self.num_pairs_in_total >= self.num_pairs_at_most:
            li_to_use = Random(self.epoch_cnt).sample(
                self.data_li, self.num_pairs_at_most
            )
            unique_ids = []
            for pair in li_to_use:
                u1, u2 = pair[0].unique_id, pair[1].unique_id
                unique_ids.append((u1, u2))
            # unique_ids = [pair[0].unqiue_id for pair in li_to_use]
            # saver.log_info(
            #     f'self.num_pairs_in_total={self.num_pairs_in_total} >= self.num_pairs_at_most={self.num_pairs_at_most}; need to do random subsampling; self.epoch_cnt={self.epoch_cnt}; after random subsampling, len(li_to_use)={len(li_to_use)}; unique_ids[0:10]={unique_ids[0:10]=}'
            # )
            # return pair_dict
        else:
            li_to_use = self.data_li

        if FLAGS.subtask == 'train' and FLAGS.symmetry_aug:
            li_to_use_aug = []
            for pair in li_to_use:
                assert len(pair) == 4
                li_to_use_aug.append((pair[1], pair[0], pair[3], pair[2]))
            li_to_use_aug = li_to_use + li_to_use_aug
            assert len(li_to_use_aug) == 2 * len(li_to_use)

            saver.log_info_once(f'symmetry_aug: li_to_use_aug: {len(li_to_use)} --> {len(li_to_use_aug)}')
            li_to_use = li_to_use_aug

        return li_to_use


class PairwiseDataset(Dataset):
    def __init__(self, generator, correct_num_data):
        self.generator = generator
        self.len_ = _get_num_iters_total(correct_num_data)
        
        self.current_index = 0
        self.num_designs = correct_num_data * 2

    def __getitem__(self, index):
        if index >= self.len_:
            raise IndexError("Index out of range")

        while self.current_index <= index:
            self.current_item = next(self.generator, None)
            self.current_index += 1

        return self.current_item

    def __len__(self):
        return self.len_

    def get_num_designs(self):
        return self.num_designs


def pairwise_custom_collate(data_list):
    if isinstance(data_list[0], Batch):
        rtn = Batch.from_data_list(data_list)#.to(FLAGS.device)
        # rtn.num_graphs_ = sum([d.num_graphs for d in data_list])
        return rtn
    else:
        raise RuntimeError(f'type(data_list){type(data_list)}; data_list={data_list}')
        return data_list


def _get_num_iters_total(num_data):
    return math.ceil(num_data / FLAGS.pairwise_batch_size)


def _get_pairwise_data_batch_yield(li, tvt):
    '''
    Cached results in subsequent epochs?
    '''

    num_iters_total = _get_num_iters_total(len(li))

    # saver.log_info(
    #     f'Pairwise data loader {tvt}: {len(li)} data points (// {FLAGS.pairwise_batch_size} --> {num_iters_total} iters); Generating Batch'
    # )

    for iter_id in tqdm(range(num_iters_total)):

        begin = iter_id * FLAGS.pairwise_batch_size
        end = min(begin + FLAGS.pairwise_batch_size, len(li))
        chunk = li[begin:end]
        individual_data_li = []
        for p in chunk:
            individual_data_li.append(p[0])  # critical: first/top half --> d_1
        for p in chunk:
            individual_data_li.append(p[1])  # critical: second/buttom half --> d_2
        if iter_id != num_iters_total - 1:
            assert len(individual_data_li) == 2 * FLAGS.pairwise_batch_size
        batch = Batch.from_data_list(individual_data_li)

        # batch.num_graphs = len(individual_data_li)

        # saver.log_info_once(f'pairwise batch.x.device={batch.x.device}')
        yield batch


def _deduce_num_data(pragma_differing_by, li):
    if type(pragma_differing_by) is tuple:
        assert len(pragma_differing_by) == 3
        num_pairs_at_most = pragma_differing_by[2]
        if num_pairs_at_most == -1:
            num_pairs_at_most = float('inf')
        pragma_differing_by = pragma_differing_by[0]
        # saver.log_info(f'num_pairs_at_most={num_pairs_at_most}')
    else:
        num_pairs_at_most = float('inf')

    num_pairs_in_total = len(li)

    correct_num_data = min(num_pairs_at_most, num_pairs_in_total)
    return num_pairs_at_most, num_pairs_in_total, correct_num_data


def gather_eval_pair_data(
    data_dict,
    points_pred_by_gname=None,
    pairs_pred_by_gname=None,
    target_list=None,
    test_name=None,
    # save_pair_li_as_csv=True,
    pragma_differing_by=1,
):

    if type(pragma_differing_by) is tuple:
        assert len(pragma_differing_by) == 3
        pragma_differing_by = pragma_differing_by[0]
    assert pragma_differing_by in [
        1,
        2,
        3,
        '>=4',
        'all',
    ], f'Only support these so far yet pragma_differing_by={pragma_differing_by}'
    # data_dict = defaultdict(list)
    # for i, data in enumerate(tqdm(dataset)):
    #     # if i == 10:
    #     #     break  # TODO: uncomment for debugging
    #     data_dict[data.gname].append(data)

    pair_dict = {}
    tlist_for_acc = ['all']
    pred_dict_by_target_global = create_pred_dict(tlist_for_acc, extra_entries=['name'])
    # pairwise_prompts_data = []

    # pairwise_stats_recorder = defaultdict(list)
    pair_dict = _looping_data_to_gen_pairs(
        data_dict,
        points_pred_by_gname,
        pairs_pred_by_gname,
        pragma_differing_by,
        pred_dict_by_target_global,
        pair_dict,
        target_list,
    )
    # for stat_label, li in pairwise_stats_recorder.items():
    # print_stats(li, stat_label, saver=saver)

    if test_name is not None:
        saver.log_info(
            f"len(pred_dict_by_target_global['all']['pred'])={len(pred_dict_by_target_global['all']['pred'])}"
        )
        saver.save_dict(
            pred_dict_by_target_global, f'{test_name}_pred_dict_by_target_global.pkl'
        )

        if len(pred_dict_by_target_global['all']['pred']) == 0:
            msg = f'Empty pred_dict_by_target_global!'
            if FLAGS.DEBUG:
                saver.log_info(msg)
            else:
                raise RuntimeError(msg)

    return pair_dict, pred_dict_by_target_global


def _looping_data_to_gen_pairs(
    data_dict,
    points_pred_by_gname,
    pairs_pred_by_gname,
    pragma_differing_by,
    pred_dict_by_target_global,
    pair_dict,
    target_list,
):
    '''
    Triple loop. Loop through kernels (gname), and then loop through design pairs.
    '''
    num_pairs_in_total = 0

    for gname, data_li in tqdm(sorted(data_dict.items())):

        # saver.log_info('Here@@@@@!!!!!!')

        if points_pred_by_gname is not None and len(points_pred_by_gname) > 0:
            if gname not in points_pred_by_gname:
                # saver.log_info('ski[]')
                continue  # no need to loop through all kernels
        if pairs_pred_by_gname is not None and len(pairs_pred_by_gname) > 0:
            if gname not in pairs_pred_by_gname:
                continue  # no need to loop through all kernels

        seen_designs = set()  # TODO

        # saver.log_info('Here@@@@@')
        # exit()

        # pred_dict_by_target_local = create_pred_dict(tlist_for_acc)
        pair_li = []
        for i, data1 in enumerate(data_li):
            for j, data2 in enumerate(data_li):

                assert data1.gname == data2.gname
                if i < j:  # only check triangle
                    d1 = data1.xy_dict_programl['point']
                    d2 = data2.xy_dict_programl['point']
                    if _check_dict_diff_by(eval(d1), eval(d2), pragma_differing_by):
                        if points_pred_by_gname:
                            saver.log_info_once(
                                f'gather_eval_pair_data: points_pred_by_gname'
                            )
                            assert not pairs_pred_by_gname
                            _add_empty_li_if_not_exsits(
                                pred_dict_by_target_global['all'], 'emb_diff'
                            )

                            pred_1 = points_pred_by_gname[gname].get(d1)
                            pred_2 = points_pred_by_gname[gname].get(d2)

                            if pred_1 is not None and pred_2 is not None:
                                for t in target_list:

                                    # saver.log_info(f'target_list={target_list}\npred_1={pred_1}\npred_2={pred_2}')
                                    # exit()
                                    pred_comp, true_comp = _pairwise_get_pred(
                                        pred_1, pred_2, data1, data2, t
                                    )
                                    # pred_dict_by_target_local['all']['pred'].append(pred_comp)
                                    # pred_dict_by_target_local['all']['true'].append(true_comp)
                                    pred_dict_by_target_global['all']['pred'].append(
                                        pred_comp
                                    )
                                    pred_dict_by_target_global['all']['true'].append(
                                        true_comp
                                    )
                                    pred_dict_by_target_global['all']['name'].append(
                                        (gname, d1, d2)
                                    )

                                # TODO: enable it
                                # emb_T_1 = pred_1['emb_T']
                                # emb_T_2 = pred_2['emb_T']
                                # emb_diff = distance.euclidean(emb_T_1, emb_T_2)

                                # pred_dict_by_target_global['all']['emb_diff'].append(emb_diff)

                        elif pairs_pred_by_gname:
                            saver.log_info_once(
                                f'gather_eval_pair_data: pairs_pred_by_gname'
                            )
                            pred_1 = None
                            pred_2 = None

                            assert not points_pred_by_gname
                            pred_dict = pairs_pred_by_gname[gname].get((d1, d2))
                            if pred_dict is not None:
                                for t in target_list:
                                    pred_dict = pred_dict['comp_label']
                                    pred_val = pred_dict[t]
                                    assert (
                                        pred_val == 0 or pred_val == 1
                                    ), f'pred_val={pred_val}\npred_dict={pred_dict}'
                                    pred_comp = pred_val
                                    true_comp = _get_comp_result_data(data1, data2, t)
                                    pred_dict_by_target_global['all']['pred'].append(
                                        pred_comp
                                    )
                                    pred_dict_by_target_global['all']['true'].append(
                                        true_comp
                                    )
                                    pred_dict_by_target_global['all']['name'].append(
                                        (gname, d1, d2)
                                    )

                        else:
                            pred_1, pred_2 = None, None

                        # Add textual repr to data so that GPT can process.

                        if FLAGS.sequence_modeling:
                            if not hasattr(data1, 'txt'):
                                data_1_txt = _gen_text_repr_of_design(data1)
                                data1.txt = data_1_txt
                            if not hasattr(data2, 'txt'):
                                data_2_txt = _gen_text_repr_of_design(data2)
                                data2.txt = data_2_txt

                        pair_li.append((data1, data2, pred_1, pred_2))
                        num_pairs_in_total += 1
                        seen_designs.add(i)
                        seen_designs.add(j)

        # if len(pred_dict_by_target_local) > 0:
        #     _report_class_result(pred_dict_by_target_local, f'{gname}_pairwise_pred_dict_by_target_local:')

        # ll = len(pair_li)
        # tot = len(data_li) * len(data_li)
        # saver.log_info(f'{gname}: Found {ll} pairs out of {len(data_li)}*{len(data_li)}={tot} pairs'
        #                f' ({ll}/{tot}={ll / tot:.2%})'
        #                f' -- seen_designs {len(seen_designs)}/{len(data_li)}={len(seen_designs) / len(data_li):.2%}', silent=True)
        pair_dict[gname] = pair_li

        # if save_pair_li_as_csv:
        #     _save_pair_li_as_csv(
        #         gname,
        #         pair_li,
        #         pairwise_prompts_data,
        #         pairwise_stats_recorder,
        #         target_list,
        #     )

    return pair_dict


def get_data_dict_by_gname(dataset):
    data_dict = defaultdict(list)

    # gname_file_li_map_fn = join(dataset.save_dir, f'gname_file_li_map_fn.pkl')
    # gname_file_li_map = load_pickle(gname_file_li_map_fn)
    # if gname_file_li_map is None:
    #     gname_file_li_map = defaultdict(list)
    #     for file in tqdm(file_li):
    #         data = torch.load(file)
    #         file_gname_map[file] = data.gname
    #     save_pickle(file_gname_map, file_gname_map_fn)
    # return

    saver.log_info(
        f'Going through all {len(dataset)} data points which will take a while'
    )

    for i, file in enumerate(tqdm(dataset.processed_file_names)):
        data = torch.load(file)  # .to(FLAGS.device)
        # if i == 100:
        #     break  # TODO: uncomment for debugging

        # assign a unique ID
        data.unique_id = i

        data_dict[data.gname].append(data)

    # below: slow
    # for i, data in enumerate(tqdm(dataset)):  # takes a while; not too fast; need to load one by one
    #     # if i == 100:
    #     #     break  # TODO: uncomment for debugging
    #     data_dict[data.gname].append(data)
    assert len(data_dict) > 0
    return data_dict


def _check_dict_diff_by(d1, d2, pragma_differing_by):
    diff_count = 0
    for k1, v1 in d1.items():
        if d2[k1] != v1:
            diff_count += 1
    if diff_count == 0:
        saver.log_info(f'Warning! d1 and d2 are identical: d1={d1}\nd2={d2}')

    if pragma_differing_by == 1:
        return diff_count == 1
    elif pragma_differing_by == 2:
        return diff_count == 2
    elif pragma_differing_by == 3:
        return diff_count == 3
    elif pragma_differing_by == '>=4':
        return diff_count >= 4
    elif pragma_differing_by == 'all':
        return True  # ie any pair is fine
    else:
        assert False
    # return True


def _save_pair_li_as_csv(
    gname,
    pair_li,
    # pairwise_prompts_data,
    pairwise_stats_recorder=None,
    target_list=None,
):
    if target_list is None:
        from data import TARGETS

        target_list = TARGETS
    fn = join(saver.get_obj_dir(), f"{gname}_pair_data.pkl")
    record_li = []
    num_tokenized = 0
    for data1, data2, pred1, pred2 in pair_li:
        repr = OrderedDict()
        # for did, data in [(1, data1), (2, data2)]:
        #     for k, v in eval(data.xy_dict_programl['point']).items():
        #         repr[f'{did}_{k}'] = v

        for t in target_list:
            for did, data in [(1, data1), (2, data2)]:
                repr[f'{did}_{t}_true'] = data.xy_dict_programl[t].item()

        for t in target_list:
            for did, pred in [(1, pred1), (2, pred2)]:
                if pred is not None:
                    pred_val = pred[t]
                    repr[f'{did}_{t}_pred'] = pred_val
                else:
                    pred_val = None

        perf_true_comp = None
        for t in target_list:
            true_comp = _get_comp_result_data(data1, data2, t)
            if t == 'perf':
                perf_true_comp = true_comp
            repr[f'{t} 1<=2? true'] = true_comp
            if pred1 is not None and pred2 is not None:
                pred_comp, _ = _pairwise_get_pred(pred1, pred2, data1, data2, t)
                repr[f'{t} 1<=2? pred'] = pred_comp
        assert perf_true_comp is not None
        if perf_true_comp == 1:
            # This means data 1's perf is < data 2's perf
            # i.e. data 1 has a larger latency,
            # i.e. answer is Yes
            answer = 'Yes'
        else:
            assert perf_true_comp == 0
            # answer is no
            answer = 'No'

        record_li.append(repr)

        if FLAGS.sequence_modeling:
            data_1_txt = _gen_text_repr_of_design(data1)
            data_2_txt = _gen_text_repr_of_design(data2)

            # Be super careful! See comment above _get_comp_result() to see why we are asking **higher** latency!
            prompt = f'#design 1:\n{data_1_txt}\n#design 2:\n{data_2_txt}\nDoes design 1 have a higher latency that design 2? [Yes/No]'
            # pairwise_prompts_data.append((prompt, answer))
            saver.log_info_at_most(f'---\n{prompt}---\n', 'prompt', 1)

            if num_tokenized == 0:
                num_tokens = len(tokenizer.tokenize(prompt))
                num_tokenized += 1
                if pairwise_stats_recorder is not None:
                    pairwise_stats_recorder['num_tokens'].append(num_tokens)

            if pairwise_stats_recorder is not None:
                pairwise_stats_recorder['pair_cnt'].append(1)
            repr['prompt'] = prompt
            repr['answer'] = answer

            assert data1.gname == data2.gname
            repr['gname'] = data1.gname

    if pairwise_stats_recorder is not None:
        pairwise_stats_recorder['num_pairs_in_kernel'].append(len(record_li))

    if FLAGS.sequence_modeling:
        df = pd.DataFrame.from_records(record_li)
        df.to_pickle(fn)
        saver.log_info(f'Saved csv to {fn} with {len(record_li)} rows')
    else:
        saver.log_info(f'NOT sequence_modeling! Nothing saved.')


def _gen_text_repr_of_design(data):
    g = read_source_code(gexf_file=None, gname=data.gname, print_msg=False)
    _, text_to_use, _ = get_text_representation_of_design(
        g, eval(data.xy_dict_programl['point'])
    )
    # print(text_to_use)
    # exit()
    return text_to_use


def _pairwise_get_pred(pred_1, pred_2, data1, data2, t):
    pred_comp = _get_comp_result(pred_1[t], pred_2[t])
    true_comp = _get_comp_result_data(data1, data2, t)
    return pred_comp, true_comp


def _get_comp_result_data(data1, data2, t):
    return _get_comp_result(
        data1.xy_dict_programl[t].item(), data2.xy_dict_programl[t].item()
    )


# Below code is super critical!
# Essentially we are checking if design 1's <target> is <= design 2's.
# In natrual lnaguage, this is asking, 'Does design 1 have a higher latency (i.e. lower perf) than design 2?'
def _get_comp_result(e1, e2):
    return 1 if e1 <= e2 else 0  # TODO: double check


# If model predicts [0, 1] for (design_1, design_2), it means the model thinks that
# design_2 is better.
# If model predicts [1, 0] for (design_1, design_2), it means the model thinks that
# design_1 is better.
def get_comp_result_tagret(y1, y2):
    return (y1 <= y2).long().flatten()  # TODO: consistent with above: 1 if <=


def _add_empty_li_if_not_exsits(d, key):
    assert type(d) is dict
    if key not in d:
        d[key] = []


############################### Ranking related ###############################

import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
import torch.nn.functional as F
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np

KENDALL_TAU = 'Kendall\'s Tau'
PRECISION_AT_10 = 'Precision@10'
RECALL_AT_10 = 'Recall@10'
MRR = "mean_reciprocal_rank"
NDCG_AT_10 = "ndcg_at_10"
MAP_SCORE = "map"
SPEARMAN_RHO = 'spearman_rho'


RANKING_METRICS = [KENDALL_TAU, PRECISION_AT_10, RECALL_AT_10, NDCG_AT_10, MRR, MAP_SCORE, SPEARMAN_RHO]


def _calculate_ranking_metrics(true_ranking, pred_ranking):
    tau, _ = kendalltau(true_ranking, pred_ranking)
    rho, _ = spearmanr(true_ranking, pred_ranking)
    precision_at_10 = len(set(true_ranking[:10]) & set(pred_ranking[:10])) / 10
    recall_at_10 = len(set(true_ranking[:10]) & set(pred_ranking[:10])) / len(
        true_ranking[:10]
    )

    reciprocal_rank = 0
    for i, item in enumerate(pred_ranking):
        if item in true_ranking:
            reciprocal_rank = 1 / (i + 1)
            break
    mrr = reciprocal_rank

    # Calculate NDCG@10 (you can adjust the k value as needed)
    true_relevance = [1 if item in true_ranking else 0 for item in pred_ranking]
    ndcg = ndcg_score([true_relevance], [np.ones(len(true_relevance))], k=10)

    # Calculate Mean Average Precision (MAP)
    # true_relevance = [1 if item in true_ranking else 0 for item in pred_ranking]
    map_score = average_precision_score([true_relevance], [np.ones(len(true_relevance))])


    return {
        KENDALL_TAU: tau,
        SPEARMAN_RHO: rho,
        PRECISION_AT_10: precision_at_10,
        RECALL_AT_10: recall_at_10,
        MRR: mrr,
        NDCG_AT_10: ndcg,
        MAP_SCORE: map_score,
        "support": len(true_ranking),
    }


def evaluate_ranking_performance(data_dict, points_pred_by_gname, pairs_pred_by_gname):
    ranking_metrics = {}
    obj_dict = {}

    for kernel_name, data_li in tqdm(sorted(data_dict.items())):


        if points_pred_by_gname is not None and len(points_pred_by_gname) > 0:
            if kernel_name not in points_pred_by_gname or len(points_pred_by_gname[kernel_name]) == 0:
                continue  # no need to loop through all kernels
        if pairs_pred_by_gname is not None and len(pairs_pred_by_gname) > 0:
            if kernel_name not in pairs_pred_by_gname or len(pairs_pred_by_gname[kernel_name]) == 0:
                continue  # no need to loop through all kernels



        # saver.log_info(f'kernel_name={kernel_name}')

        if points_pred_by_gname:
            assert not pairs_pred_by_gname

            # saver.log_info(f'points_pred_by_gname[kernel_name]={points_pred_by_gname[kernel_name]}')
            pred_designs = set(points_pred_by_gname[kernel_name].keys())

            # saver.log_info(f'points_pred_by_gname; pred_designs={pred_designs}')

        elif pairs_pred_by_gname:
            assert not points_pred_by_gname

            pred_designs = set(
                design
                for pair in pairs_pred_by_gname[kernel_name].keys()
                for design in pair
            )

            # saver.log_info(f'pairs_pred_by_gname; pred_designs={pred_designs}')

        else:
            assert False

        # for kernel_name, design_pairs in pairs_pred_by_gname.items():

        true_designs =  set(
            data.xy_dict_programl['point'] for data in data_li
        )

        designs_in_both = true_designs & pred_designs

        # saver.log_info(f'true_designs={true_designs}')
        # saver.log_info(f'designs_in_both={designs_in_both}')

        true_perfs = {
            data.xy_dict_programl['point']: data.xy_dict_programl["perf"].item()
            for data in data_li
            if data.xy_dict_programl['point'] in designs_in_both
        }
        true_ranking = sorted(true_perfs, key=true_perfs.get, reverse=True)

        pred_scores = {}

        if points_pred_by_gname:

            for d, pred in points_pred_by_gname[kernel_name].items():
                if d in designs_in_both:
                    # Just use the predicted perf as the score.
                    # The larger the better.
                    # pred_scores[d] = pred_scores.get(d, 0) + pred['perf']
                    assert d not in pred_scores
                    pred_scores[d] = pred['perf']


        elif pairs_pred_by_gname:
            for (d1, d2), pred in pairs_pred_by_gname[kernel_name].items():
                if d1 in designs_in_both and d2 in designs_in_both:
                    comp_logits = pred['comp_logits']

                    # comp_probs = F.softmax(comp_logits)
                    comp_probs = convert_out_logits_to_rank_scores(comp_logits)
                    # Use the sottmax prob as the score.
                    assert comp_probs.shape == (1,2)
                    pred_scores[d1] = pred_scores.get(d1, 0) + comp_probs[0][0]
                    pred_scores[d2] = pred_scores.get(d2, 0) + comp_probs[0][1]
                    # TODO: write some assert to check each design is updated equal # of times
                    # TODO: This is to avoid the following scnenario: (1, 3), (1, 5) are compared yet design 1 is updated twice... unfair to other designs...`
        else:
            assert False

        # for (d1, d2), pred in design_pairs.items():
        #     if d1 in designs_in_both and d2 in designs_in_both:
        #         comp_logits = pred['comp_logits']

        #         comp_probs = F.softmax(comp_logits)
        #         pred_scores[d1] = pred_scores.get(d1, 0) + comp_probs[0]
        #         pred_scores[d2] = pred_scores.get(d2, 0) + comp_probs[1]

        pred_ranking = sorted(pred_scores, key=pred_scores.get, reverse=True)

        assert len(designs_in_both) == len(true_ranking) == len(pred_ranking)
        ranking_metrics[kernel_name] = _calculate_ranking_metrics(
            true_ranking, pred_ranking
        )

        obj_dict[kernel_name] = {
            'true_ranking': true_ranking,
            'pred_ranking': pred_ranking,
        }

    best_kernel = max(ranking_metrics, key=lambda k: ranking_metrics[k][KENDALL_TAU])
    worst_kernel = min(ranking_metrics, key=lambda k: ranking_metrics[k][KENDALL_TAU])
    _plot_ranking_results(
        f'{best_kernel}_best',
        obj_dict[best_kernel]['true_ranking'],
        obj_dict[best_kernel]['pred_ranking'],
    )
    _plot_ranking_results(
        f'{worst_kernel}_worst',
        obj_dict[worst_kernel]['true_ranking'],
        obj_dict[worst_kernel]['pred_ranking'],
    )

    _print_ranking_metrics_table(ranking_metrics)
    ranking_metrics_aggr = _aggregate_ranking_metrics(ranking_metrics)
    return ranking_metrics_aggr


def _print_ranking_metrics_table(ranking_metrics):
    header = f'| {"Kernel Name":<20}| {"KENDALL_TAU":>13} | {"PRECISION_AT_10":>16} |' \
            f' {"RECALL_AT_10":>13} | {"Design Count":>12} |'

    separator = '|---------------------|---------------|-------------------|---------------|--------------|'

    table_rows = [header, separator]

    for kernel_name, metrics in ranking_metrics.items():
        tau = metrics[KENDALL_TAU]
        p = metrics[PRECISION_AT_10]
        r = metrics[RECALL_AT_10]
        s = metrics["support"]
        row = (
            f'| {kernel_name:<20}| {tau:>13.3f} | {p:>16.3f} |'
            f' {r:>13.3f} | {s:>12} |'
        )
        table_rows.append(row)

    table = '\n'.join(table_rows)
    saver.log_info(f'\nRanking Metrics Table:\n{table}\n')


def _plot_ranking_results(kernel_name, true_ranking, pred_ranking):
    plt.figure(figsize=(10, 6))

    # Convert true_ranking to a list of strings
    true_ranking_str = [str(design) for design in true_ranking]

    # Plot true ranking
    plt.plot(
        range(len(true_ranking_str)), range(len(true_ranking_str)), label='True Ranking'
    )

    # Plot predicted ranking
    pred_ranking_positions = [pred_ranking.index(design) for design in true_ranking_str]
    plt.plot(
        range(len(true_ranking_str)), pred_ranking_positions, label='Predicted Ranking'
    )

    # Highlight top N designs
    top_N = min(10, len(true_ranking_str))
    top_N_positions = [
        pred_ranking.index(design) for design in true_ranking_str[:top_N]
    ]
    plt.scatter(
        range(top_N),
        top_N_positions,
        marker='o',
        s=100,
        label=f'Top {top_N} True Designs',
    )

    plt.xlabel('True Ranking Position')
    plt.ylabel('Predicted Ranking Position')
    plt.title(f'Ranking Results for Kernel: {kernel_name}')
    plt.legend()

    sp = join(saver.get_plot_dir(), f'ranking_results_{kernel_name}.png')
    plt.savefig(sp)
    saver.log_info(sp)
    plt.close()


def _aggregate_ranking_metrics(ranking_metrics):
    aggregated_metrics = {}
    for metric_name in ranking_metrics[next(iter(ranking_metrics))].keys():
        metric_values = [metrics[metric_name] for metrics in ranking_metrics.values()]
        aggregated_metrics[metric_name] = sum(metric_values) / len(metric_values)

    return aggregated_metrics




from node_att_diff import split_vec_mat_into_2_halves
import torch.nn as nn



class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        if 'cross_entropy' in FLAGS.pairwise_loss:
            self.xentropy_loss = nn.CrossEntropyLoss()
        if 'mse' in FLAGS.pairwise_loss or 'padr' in FLAGS.pairwise_loss:
            self.mse_loss = nn.MSELoss(reduction='mean')

        assert FLAGS.pairwise_loss in ['None', 'cross_entropy', 'ranknet_v1', 'ranknet_v2', 'ranknet_v1+mse', 'ranknet_v2+mse', 'cross_entropy+mse', 'padr']

        assert FLAGS.pairwise_loss_mul_factor >= 0
        
            

    def forward(self, out, target, loss_dict, strict_even):

        y1, y2 = split_vec_mat_into_2_halves(target, strict_even=strict_even)
        loss = 0.0

        if 'cross_entropy' in FLAGS.pairwise_loss:
            target_pairwise = get_comp_result_tagret(y1, y2)
            # Tricky: y1 <= y2
            # so 1 if y1 is worse, 0 if y1 is better
            # i.e. logits/out means [design_1 is better, design_2 is better]
            assert out.shape[0] == target_pairwise.shape[0]
            rank_loss = self.xentropy_loss(out, target_pairwise)

            loss_dict['cross_entropy_in_pairwise'] = rank_loss

        elif 'ranknet' in FLAGS.pairwise_loss:
            preds_diff = out[:, 0] - out[:, 1]
            # Tricky: interprete the logits/out differently from above
            # logits/out means [design_1's score/perf (bigger is better), design_2's score/perf (bigger is better)]
            # However, intuitvely this still means [design_1 is better, design_2 is better]
            

            if 'ranknet_v1' in FLAGS.pairwise_loss:
                # Calculate the probability using the logistic function
                prob = 1.0 / (1.0 + torch.exp(-preds_diff))

                target_pairwise = (y1 > y2).float()  # Convert boolean to float for calculation
                saver.log_info_at_most(f'ranknet_v1:target_pairwise={target_pairwise}\nout.shape={out.shape}\ntarget.shape={target.shape}', 'target_pairwise_ranknet_v1', 1)
                # Tricky: y1 > y2
                # so 1 if y1 is better, 0 is y2 is better
                # Since logits/out means []design_1's score/perf (o1), design_2's score/perf (02)]
                # prob means sigmoid(o1 - o2)
                # so when y1 is better,
                # L = -(1 * log(sigmoid(o1 - o2))) 
                #   = -log(sigmoid(o1 - o2))
                #   --> encourages o1 is to bigger than o2
                # when y2 is better
                # L = -(1 * log(1 - sigmoid(o1 - o2))) 
                #   = -log(1 - sigmoid(o1 - o2))
                #   --> encourages o1 is to smaller than o2

                rank_loss = -torch.mean(target_pairwise * torch.log(prob + 1e-10) + (1 - target_pairwise) * torch.log(1 - prob + 1e-10))
                loss_dict['ranknet_v1_loss_in_pairwise'] = rank_loss

            elif 'ranknet_v2' in FLAGS.pairwise_loss:
                # preds_diff = out[:, 0] - out[:, 1]
                target_diff = y1 - y2
                rank_loss = torch.mean(1.0 - torch.sigmoid(target_diff * preds_diff))

                loss_dict['ranknet_v2_loss_in_pairwise'] = rank_loss
            else:
                assert False

        elif 'padr' in FLAGS.pairwise_loss:
            preds_diff = out[:, 0] - out[:, 1]
            target_diff = y1 - y2
            rank_loss = self.mse_loss(preds_diff, target_diff)

            loss_dict['padr_loss_in_pairwise'] = rank_loss


        loss += rank_loss

        if 'mse' in FLAGS.pairwise_loss:
            # Tricky code: out is the split design-level embeddings in model.py
            # so here need to stack back.
            new_out = out.view(-1, 1)


            mse_loss = self.mse_loss(new_out, target)
            loss += mse_loss

            loss_dict['mse_loss_in_pairwise'] = mse_loss


        return FLAGS.pairwise_loss_mul_factor * loss

                


def convert_out_logits_to_rank_scores(out_logits):
    if FLAGS.pairwise_loss == 'cross_entropy':
        rank_scores = F.softmax(out_logits, dim=1)
        if FLAGS.rank_score_deduction_cross_entropy == 'softmax':
            pass
        elif FLAGS.rank_score_deduction_cross_entropy == 'softmax_0.5_to_hard':
            rank_scores = (rank_scores > 0.5).float()   # hardened
        else:
            assert False
    elif 'ranknet' in FLAGS.pairwise_loss or 'padr' in FLAGS.pairwise_loss:
        rank_scores = out_logits # raw scores mean what the model thinks should assign to the designs
    else:
        assert False

    saver.log_info_at_most(f'convert_out_logits_to_rank_scores:out_logits={out_logits}\n-->\nrank_scores={rank_scores}', 'convert_out_logits_to_rank_scores', 1)
    return rank_scores

