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
# Copyright (c) 2023, UCLA VAST Lab (HARP)
# BSD 3-Clause License (see LICENSE-THIRD-PARTY for full text)
#
# Portions provided under the following terms:
# Copyright (c) UCLA-DM (HLSyn)
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)
# (see LICENSE-THIRD-PARTY for full text)

import config
from database import create_database
from data_code2vec import (
    init_preprocessors_code2vec,
    read_code2vec_data,
    encode_feat_dict_code2vec,
    fit_preprocessors_code2vec,
    encode_X_torch_code2vec,
)
from data_multi_modality import (
    init_preprocessors_multi_modality,
    read_multi_modality_data,
    encode_feat_dict_multi_modality,
    fit_preprocessors_multi_modality,
    encode_X_torch_multi_modality,
)
from data_programl import (
    init_preprocessors_programl,
    read_programl_graph,
    encode_feat_dict_programl,
    fit_preprocessors_programl,
    encode_X_torch_programl,
)
from data_src_code import (
    init_preprocessors_src_code,
    read_source_code,
    encode_feat_dict_src_code,
    fit_preprocessors_src_code,
    encode_X_torch_src_code,
)
from graph_transfotmer import collate_batch_graph_transformer


from config import FLAGS
from saver import saver
from utils import (
    get_root_path,
    MLP,
    print_stats,
    get_save_path,
    create_dir_if_not_exists,
    plot_dist,
    save_pickle,
    load_pickle,
    load,
    create_edge_index,
    sorted_nicely,
    coo_to_sparse,
    save,
    OurTimer,
    report_save_dir,
)
from result import Result, persist

from os.path import join, basename, dirname
from glob import glob, iglob

from math import ceil

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

import redis
import pickle
import random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm

import os.path as osp

import torch
from torch_geometric.data import Dataset

from shutil import rmtree
from torch.utils.data import random_split
import math
import os
import time

KEY_PRAGMA = '__PIPE__L0'
FURTHER_SPLIT = ''
if FLAGS.subtask == 'train' and FLAGS.MAML:
    if FLAGS.MAML_split_kernel:
        FURTHER_SPLIT = f'split_{KEY_PRAGMA}'
    else:
        FURTHER_SPLIT = f'nosplit'
else:
    FURTHER_SPLIT = f'nosplit'


TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']


def _get_save_dir(FLAGS):
    if hasattr(FLAGS, 'force_data_dir') and FLAGS.force_data_dir is not None:
        saver.log_info(f'FLAGS has force_data_dir; just use that')
        return FLAGS.force_data_dir

    if FLAGS.encoder_path is not None and FLAGS.encoder_path != 'None':
        encoder_str = FLAGS.load_encoders_label
    else:
        encoder_str = 'regular_encoder'
        # encoder_str = 'None'

    need_shrinking = False
    if FLAGS.v_db == 'v20':
        if FLAGS.only_common_db:
            if FLAGS.test_extra:
                SAVE_DIR = join(
                    get_save_path(),
                    FLAGS.dataset,
                    f'{FLAGS.v_db}_with-updated-task-transfer-new-db-new-speedup-common-only-extra-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                )
            else:
                SAVE_DIR = join(
                    get_save_path(),
                    FLAGS.dataset,
                    f'{FLAGS.v_db}_with-updated-task-transfer-new-db-new-speedup-common-only-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                )
        else:
            if FLAGS.encoder_path is None:
                SAVE_DIR = join(
                    get_save_path(),
                    FLAGS.dataset,
                    f'{FLAGS.v_db}_with-updated-new-db-new-speedup-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                )
            else:
                if not FLAGS.only_new_points:
                    SAVE_DIR = join(
                        get_save_path(),
                        FLAGS.dataset,
                        f'{FLAGS.v_db}_v2-transfer-round{FLAGS.round_num}-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                    )
                else:
                    SAVE_DIR = join(
                        get_save_path(),
                        FLAGS.dataset,
                        f'{FLAGS.v_db}_with-updated-task-transfer-dse1-only-new-db-new-speedup-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                        'dse3',
                    )
    elif FLAGS.v_db == 'v18':
        if FLAGS.new_speedup:
            # SAVE_DIR = join(get_save_path(), FLAGS.dataset,
            #                 f'round{FLAGS.round_num}-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            SAVE_DIR = join(
                get_save_path(),
                FLAGS.dataset,
                f'{FLAGS.v_db}_r{FLAGS.round_num}-ifdb-{FLAGS.graph_type}-{FLAGS.task}_ep-{FLAGS.encode_edge_position}_nowi_{FLAGS.invalid}-n_{FLAGS.norm_method}_np_{FLAGS.no_pragma}_{FLAGS.tag}',
            )
        else:
            # SAVE_DIR = join(get_save_path(), FLAGS.dataset,
            #                 f'with-updated-task-transfer-old-db-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            SAVE_DIR = join(
                get_save_path(),
                FLAGS.dataset,
                f'{FLAGS.v_db}_wutfod-{FLAGS.task}_wi_{FLAGS.invalid}-n_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_{FLAGS.tag}',
            )
    elif FLAGS.v_db == 'v21':
        SAVE_DIR = join(
            get_save_path(),
            FLAGS.dataset,
            f'{FLAGS.v_db}_{FLAGS.pragma_as_MLP}-{FLAGS.round_num}-{FLAGS.graph_type}-{FLAGS.task}_{FLAGS.encode_edge_position}_{FLAGS.invalid}_{FLAGS.norm_method}_{FLAGS.no_pragma}_{FLAGS.tag}',
        )
    else:
        raise NotImplementedError

    if FLAGS.model == 'code2vec':
        rtn = f'{SAVE_DIR}_code2vec-{FLAGS.code2vec_data_version}'
        return rtn
    # assert FLAGS.model == 'our'

    if FLAGS.DEBUG:
        SAVE_DIR += '_DEBUG'

    ptrans_s = ''
    # if FLAGS.ptrans:
    #     ptrans_s = f'_{FLAGS.pragma_scope}'
    smodeling_s = ''
    if FLAGS.sequence_modeling:
        # smodeling_s = f'_seq_{FLAGS.code_encoder}'
        smodeling_s = f'_s_{FLAGS.data_repr}_{FLAGS.code_encoder}'
        if (
            hasattr(FLAGS, 'max_code_tokens_len')
            and FLAGS.max_code_tokens_len is not None
        ):
            smodeling_s = f'{smodeling_s}_{FLAGS.max_code_tokens_len}'
            if hasattr(FLAGS, 'bi_directional_AST') and FLAGS.bi_directional_AST:
                smodeling_s = f'{smodeling_s}_bi'
        if hasattr(FLAGS, 'token_att_masking') and FLAGS.token_att_masking:
            smodeling_s = f'{smodeling_s}_tm'
        if hasattr(FLAGS, 'token_att_masking') and FLAGS.preserve_keywords:
            smodeling_s = f'{smodeling_s}_pk'
            if hasattr(FLAGS, 'pk_version') and FLAGS.pk_version >= 2:
                smodeling_s = f'{smodeling_s}_v{FLAGS.pk_version}'
        if hasattr(FLAGS, 'add_edges_fc_graph') and FLAGS.add_edges_fc_graph:
            smodeling_s = f'{smodeling_s}_fc'
        if hasattr(FLAGS, 'chunk_offset') and FLAGS.chunk_offset != 0:
            smodeling_s = f'{smodeling_s}_co{FLAGS.chunk_offset}'

    graph_type_s = ''
    if FLAGS.graph_type != '':
        if FLAGS.graph_type not in SAVE_DIR:
            graph_type_s = f'_{FLAGS.graph_type}'

    ignore_kernels_s = ''
    if hasattr(FLAGS, 'ignore_kernels') and len(FLAGS.ignore_kernels) > 0:
        s = '_'.join([str(x) for x in FLAGS.ignore_kernels])
        ignore_kernels_s = f'_ig={s}'

    mms = ''
    if FLAGS.multi_modality:
        mms = f'_{FLAGS.what_modalities}'
        if FLAGS.combine_fashion == 'share_final_MLPs':
            if FLAGS.feed_p_to_tf:
                mms = f'{mms}_feed'
            if FLAGS.pc_links:
                # 05/05/2023: found the token_id_global_in_chunks bug (need to handle eos token) Although this only affects node_token_interaction, still change the dataset name to regenerate.
                mms = f'{mms}_pclcc'
                # mms = f'{mms}_pclc'  # 04/30/2023: c means "correct" with pragma nodes
                # mms = f'{mms}_pcl'
                need_shrinking = True
                if FLAGS.pc_links_aug is not None:
                    mms = f'{mms}_{FLAGS.pc_links_aug}'
                if FLAGS.node_token_interaction:
                    # 05/05/2023: found the token_id_global_in_chunks bug (need to handle eos token)
                    mms = f'{mms}_ntic'
                    # mms = f'{mms}_nti'
                if FLAGS.pc_links_holdout_ratio > 0:
                    mms = f'{mms}_{FLAGS.pc_links_holdout_ratio}'
                if (
                    hasattr(FLAGS, 'interleave_GNN_transformer')
                    and FLAGS.interleave_GNN_transformer
                ):
                    mms = f'{mms}_igt'
        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            need_shrinking = True
            mms = f'{mms}_{FLAGS.combine_fashion}'
            if hasattr(FLAGS, 'add_pragma_links') and FLAGS.add_pragma_links:
                mms = f'{mms}_apl'
        # elif FLAGS.combine_fashion == 'p_to_tf':
        #     need_shrinking = True
        #     mms = f'{mms}_{FLAGS.combine_fashion}'
        #     mms = f'{mms}_{FLAGS.feed_what}'
        else:
            raise NotImplementedError()

    combine_ast_programl = False  # deprecated flag
    rtn = (
        f'{SAVE_DIR}_{FLAGS.gtype}_{FLAGS.only_pragma_nodes}_{combine_ast_programl}_'
        f'{FLAGS.encode_full_text}_{FLAGS.fulltext_dim}_{FURTHER_SPLIT}_{encoder_str}_{FLAGS.pragma_as_MLP}{ptrans_s}{smodeling_s}{graph_type_s}{ignore_kernels_s}{mms}'
    )

    # if need_shrinking:
    #     len_before = len(rtn)
    #     rtn = rtn.replace('with-updated-new-db-new-speedup', 'w')
    #     rtn = rtn.replace(
    #         'with-updated-task-transfer-new-db-new-speedup-common-only-extra', 'wt')
    #     rtn = rtn.replace(
    #         'with-updated-task-transfer-dse1-only-new-db-new-speedup', 'wtd')
    #     rtn = rtn.replace('with-updated-task-transfer-old-db', 'wto')
    #     rtn = rtn.replace('normalization', 'n')
    #     saver.log_info(
    #         f'Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')

    # rtn = rtn.replace('with-updated-new-db-new-speedup', 'w')

    # if len(rtn) >= 199 or True:
    #     len_before = len(rtn)
    #     rtn = rtn.replace('v2-transfer-round2-icmp-feb-db--regression_edge-position', 'v2trifdrep')
    #     rtn = rtn.replace('v2-transfer-round2-icmp-feb-db--class_edge-position', 'v2trifdcep')
    #     rtn = rtn.replace('v2-transfer-round3-icmp-feb-db--regression_edge-position', 'v2tr3ifdrep')

    #     rtn = rtn.replace('v2-transfer-round3-icmp-feb-db--class_edge-position', 'v2tr3ifdcep')
    #     rtn = rtn.replace(
    #         'v2-transfer-round2-icmp-feb-db-extended-pseudo-block-connected-hierarchy-regression_edge-position',
    #         'v2tr3ifdcepepbchrep')

    #     rtn = rtn.replace('v2-transfer-round3-icmp-feb-db-extended-pseudo-block-connected-hierarchy-regression_edge-position', 'v2tr3ifderep')
    #     saver.log_info(f'Further Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')
    #     if len(rtn) >= 302:
    #         len_before = len(rtn)
    #         rtn = rtn.replace(
    #             'v2trifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_programl_False_False_None_None_nosplit',
    #             'v2trifdrep-s')
    #         rtn = rtn.replace(
    #             'v2trifdcep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_programl_False_False_None_None_nosplit',
    #             'v2trifdcep-s')
    #         rtn = rtn.replace(
    #             'v2tr3ifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_wmp-d_programl_False_False_None_None_nosplit',
    #             'v2trifdrep-sd')
    #         rtn = rtn.replace(
    #             'v2tr3ifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_wmp-d_programl_False_False_None_None_nosplit',
    #             'v2trifdcep-sd')
    #         saver.log_info(f'Further Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')

    # rtn = rtn.replace('with-updated-new-db-new-speedup', 'w')

    rtn = rtn.replace('extended-pseudo-block-connected-hierarchy', 'epbch')

    saver.log_info(f'len(save_dir)={len(rtn)}')
    saver.log_info(f'save_dir={rtn} ({report_save_dir(rtn)})')
    return rtn


SAVE_DIR = _get_save_dir(FLAGS)


def flush_save_dir():
    SAVE_DIR = _get_save_dir(FLAGS)


def update_save_dir(FLAGS):
    # tricky; if loaded flags are different; need to call this function to load the correct dataset
    global SAVE_DIR, ENCODER_PATH
    SAVE_DIR = _get_save_dir(FLAGS)
    ENCODER_PATH = join(SAVE_DIR, 'encoders')
    saver.log_info(
        f'SAVE_DIR and ENCODER_PATH updated due to FLAGS being potentially loaded'
    )
    saver.log_info(f'SAVE_DIR={SAVE_DIR}')


def update_save_dir_with_additional_str(additional_str):
    global SAVE_DIR, ENCODER_PATH
    SAVE_DIR = f'{SAVE_DIR}_{additional_str}'
    ENCODER_PATH = join(SAVE_DIR, 'encoders')
    saver.log_info(
        f'SAVE_DIR updated with additional str={SAVE_DIR}\nENCODER_PATH={ENCODER_PATH}'
    )


def get_save_dir_encoder_path_with_classficiation():
    global SAVE_DIR, ENCODER_PATH
    assert (
        FLAGS.task == 'regression'
    ), 'When requesting load_model_class, must set task to regression, because we assume the main task in regression and the classification task is auxiliary'
    assert 'regression' in SAVE_DIR and 'regression' in ENCODER_PATH

    if hasattr(FLAGS, 'class_model_save_dir'):
        save_dir = FLAGS.class_model_save_dir
        if (
            hasattr(FLAGS, 'encoder_path_class')
            and FLAGS.encoder_path_class is not None
        ):
            encoder_path = FLAGS.encoder_path_class
        else:
            encoder_path = join(save_dir, 'encoders')
    else:
        save_dir = SAVE_DIR.replace('regression', 'class')
        encoder_path = ENCODER_PATH.replace('regression', 'class')

    return save_dir, encoder_path


ENCODER_PATH = join(SAVE_DIR, 'encoders')
# PROCESSED_DIR = join(SAVE_DIR, 'processed')
create_dir_if_not_exists(SAVE_DIR)

# DATASET = 'cnn1'
DATASET = 'machsuite-poly'
dse_database_name = FLAGS.dse_database_name
if DATASET == 'cnn1':
    KERNEL = 'cnn'
    DB_PATHS = f'{get_root_path()}/{dse_database_name}/databases/cnn_case1/'
elif DATASET == 'machsuite':
    KERNEL = FLAGS.tag
    DB_PATHS = f'{get_root_path()}/{dse_database_name}/machsuite/databases/**/*'
elif DATASET == 'machsuite-poly':
    KERNEL = FLAGS.tag
    DB_PATHS = []
    for benchmark in FLAGS.benchmarks:
        DB_PATHS.append(
            f'{get_root_path()}/{dse_database_name}/{benchmark}/databases/**/*'
        )

GEXF_FOLDER = None


TARGETS = config.TARGETS
# MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
# poly_KERNEL = config.poly_KERNEL
# ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

GEXF_FILES = None


def update_gexf_folder_files():
    global GEXF_FOLDER, GEXF_FILES
    if FLAGS.dataset == 'vitis-cnn':
        GEXF_FOLDER = join(
            get_root_path(), dse_database_name, 'dotGenerator_all_kernels'
        )
    elif FLAGS.dataset == 'machsuite':
        GEXF_FOLDER = join(get_root_path(), dse_database_name, 'machsuite', 'dot-files')
    elif FLAGS.dataset == 'programl':
        GEXF_FOLDER = join(
            get_root_path(), dse_database_name, 'programl', '**', 'processed', '**'
        )
        if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
            GEXF_FOLDER = join(
                get_root_path(),
                dse_database_name,
                f'gexf_with_line_col_{FLAGS.graph_type}',
                '**',
            )
    elif FLAGS.dataset == 'programl-machsuite':
        GEXF_FOLDER = join(
            get_root_path(), dse_database_name, 'programl', 'machsuite', 'processed'
        )
    elif FLAGS.dataset == 'simple-programl':
        GEXF_FOLDER = join(
            get_root_path(),
            dse_database_name,
            'simple-program',
            'programl',
            'processed',
            '**',
        )
    elif FLAGS.dataset == 'harp':
        GEXF_FOLDER = join(
            get_root_path(),
            'dse_database',
            'generated_graphs',
            '**',
            'processed',
            'extended-pseudo-block-connected-hierarchy',
            '**',
        )
    elif FLAGS.dataset == 'harp-line-col':
        GEXF_FOLDER = join(get_root_path(), 'gexf_with_line_col', '**')
    else:
        raise NotImplementedError()

    if FLAGS.check_release_db:
        GEXF_FOLDER = join(get_root_path(), dse_database_name, '**')

    if FLAGS.all_kernels:
        if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
            GEXF_FILES = sorted(
                [f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')]
            )  # and 'extended' not in f])
        else:
            if FLAGS.graph_type == '':
                GEXF_FILES = sorted(
                    [
                        f
                        for f in iglob(GEXF_FOLDER, recursive=True)
                        if f.endswith('.gexf')
                        and 'extended' not in f
                        and 'processed' in f
                    ]
                )
            else:
                if 'hierarchy' not in FLAGS.graph_type:
                    pruner = 'hierarchy'
                else:
                    pruner = 'initial'
                GEXF_FILES = sorted(
                    [
                        f
                        for f in iglob(GEXF_FOLDER, recursive=True)
                        if f.endswith('.gexf')
                        and FLAGS.graph_type in f
                        and 'processed' in f
                        and pruner not in f
                    ]
                )
    else:
        GEXF_FILES = sorted(
            [
                f
                for f in iglob(GEXF_FOLDER, recursive=True)
                if f.endswith('.gexf')
                and f'{FLAGS.target_kernel}_' in f
                and 'extended' not in f
                and 'processed' in f
            ]
        )

    if FLAGS.force_regen:
        if len(GEXF_FILES) == 0:
            raise RuntimeError(f'No GEXF_FILES found. Check GEXF_FOLDER={GEXF_FOLDER}\nlist(iglob(GEXF_FOLDER, recursive=True))={list(iglob(GEXF_FOLDER, recursive=True))}')


update_gexf_folder_files()
# debugging code

# n = []
# for gexf_file in GEXF_FILES:
#     # if 'bicg' in gexf_file and 'large' not in gexf_file  and 'medium' not in gexf_file:
#     #     n.append(gexf_file)
#     if 'stencil-3d' in gexf_file:
#         n.append(gexf_file)
# GEXF_FILES = n

torch_generator = None

if FLAGS.fix_randomness:
    saver.log_info('Critical! data.py: Fix random seed for torch data loader generator')
    torch_generator = torch.Generator()
    torch_generator.manual_seed(0)


def finte_diff_as_quality(new_result: Result, ref_result: Result) -> float:
    """Compute the quality of the point by finite difference method.

    Args:
        new_result: The new result to be qualified.
        ref_result: The reference result.

    Returns:
        The quality value (negative finite differnece). Larger the better.
    """

    def quantify_util(result: Result) -> float:
        """Quantify the resource utilization to a float number.

        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'

        Args:
            result: The evaluation result.

        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """

        # Reduce the sensitivity to (100 / 5) = 20 intervals
        utils = [
            5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon
            for k, u in result.res_util.items()
            if k.startswith('util')
        ]

        # Compute the area
        return sum([2.0 ** (1.0 / (1.0 - u)) for u in utils])

    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)

    # if (new_result.perf / ref_result.perf) > 1.05:
    #     # Performance is too worse to be considered
    #     return -float('inf')

    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            # Free lunch
            # return float('inf')
            return FLAGS.max_number
        # Same util but slightly worse performance, neutral
        return 0

    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)


class MyOwnDataset(Dataset):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        data_files=None,
        need_attribs=True,
        init_feat_dict=None,
        save_dir=None,
        encoder_path=None,
    ):
        if save_dir is None:
            save_dir = SAVE_DIR
        # self.processed_dir = PROCESSED_DIR
        # print('SAVE_DIR', SAVE_DIR)
        self.save_dir = save_dir
        super(MyOwnDataset, self).__init__(save_dir, transform, pre_transform)
        self.scalers = None
        if encoder_path is None:
            encoder_path = ENCODER_PATH
        self.encoders, self.preprocessors = load_encoders_preprocessors(
            save_dir, encoder_path
        )
        if need_attribs:
            self.attribs = load_pickle(self._get_attribs_sp(), print_msg=False)
            if self.attribs is None:
                self.attribs = {}
        if data_files is not None:
            self.data_files = data_files
        if init_feat_dict is not None:
            self.add_attribute('init_feat_dict', init_feat_dict)
        if not self.has_attribute('init_feat_dict'):
            init_feat_dict = load(join(self.save_dir, 'pragma_dim'))
            self.add_attribute('init_feat_dict', init_feat_dict)
        # saver.log_info(f'MyOwnDataset init done')

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        if hasattr(self, 'data_files'):
            return self.data_files
        else:
            rtn = sorted_nicely(glob(join(self.save_dir, '*.pt')))
            return rtn

    def add_target_scalers(self, scalers, to_save):
        assert type(scalers) is dict
        self.scalers = scalers
        sp = self._get_scalers_sp()
        if to_save:
            save_pickle(scalers, sp)

    def add_attribute(self, attr_name, attr):
        if attr_name in self.attribs:
            raise RuntimeError(f'Attribute {attr_name} already exist')
        self.attribs[attr_name] = attr
        save_pickle(self.attribs, self._get_attribs_sp())

    def get_attribute(self, attr_name):
        rtn = self.attribs.get(attr_name)
        if rtn is None:
            raise RuntimeError(f'Attribute {attr_name} does not exist')
        return rtn

    def has_attribute(self, attr_name):
        return attr_name in self.attribs

    def get_target_scalers(self):
        if self.scalers is None:
            sp = self._get_scalers_sp()
            self.scalers = load_pickle(sp)
            if type(self.scalers) is not dict:
                raise ValueError(f'Cannot load scalers from {sp}')
        return self.scalers

    def _get_scalers_sp(self):
        return join(self.save_dir, 'scalers.pkl')

    def _get_attribs_sp(self):
        return join(self.save_dir, 'attribs.pkl')

    def download(self):
        pass

    # Download to `self.raw_dir`.

    def process(self):
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):

        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(self.save_dir, 'data_{}.pt'.format(idx))
        data = torch.load(fn)

        # attempts = 0
        # success = False
        # while attempts < 10:
        #     try:
        #         if hasattr(self, 'data_files'):
        #             fn = self.data_files[idx]
        #         else:
        #             fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        #         data = torch.load(fn)
        #         success = True
        #     except Exception as e:
        #         if attempts == 10:
        #             raise RuntimeError(f"{e}")
        #         attempts += 1
        #     if success == True:
        #         break
        #     time.sleep(1)

        return data


def split_dataset(dataset, batch_size=FLAGS.batch_size, concise=False):
    file_li = dataset.processed_file_names
    transductive_test_file_li = []
    if FLAGS.tvt_split_by == 'designs_transductive':
        saver.log_info(
            f'tvt_split_by {FLAGS.tvt_split_by} so just mix all designs together and split'
        )
        val_ratio = FLAGS.val_ratio
        test_ratio = FLAGS.val_ratio
        num_graphs = len(dataset)

        if FLAGS.sample_finetune:
            r1 = int(num_graphs * 1.0)
            r2 = int(num_graphs * 0)
        else:
            r1 = int(num_graphs * (1.0 - (val_ratio + test_ratio)))
            r2 = int(num_graphs * val_ratio)

        # file_li = ['xxxx'] * len(file_li)
        lengths = [r1, r2, len(dataset) - r1 - r2]
        if FLAGS.shuffle:
            li = random_split(
                file_li, lengths, generator=torch.Generator().manual_seed(100)
            )
        else:
            li = split_li(file_li, lengths)
    elif FLAGS.tvt_split_by == 'kernels_inductive':
        saver.log_info(
            f'tvt_split_by {FLAGS.tvt_split_by} so going through all data points and gather stats first'
        )
        # train_file_li = []
        val_test_kernel_li_map = defaultdict(list)
        train_kernel_li_map = defaultdict(list)
        kernel_names = set()

        # TODO: the below code should be fixed to reflect change in FLAGS.random_seed.
        # Currently. just set seed to 123.
        file_gname_map_fn = osp.join(dataset.save_dir, f'file_gname_map.pkl')
        file_gname_map = load_pickle(file_gname_map_fn)
        if file_gname_map is None:
            file_gname_map = OrderedDict()
            for file in tqdm(file_li):
                data = torch.load(file)
                file_gname_map[file] = data.gname
            save_pickle(file_gname_map, file_gname_map_fn)

        for file, gname in tqdm(file_gname_map.items()):
            # data = torch.load(file)
            if gname in FLAGS.test_kernels:
                val_test_kernel_li_map[gname].append(file)
            else:
                train_kernel_li_map[gname].append(file)
                # train_file_li.append(file)
            kernel_names.add(gname)
        if not concise:
            saver.log_info(f'-' * 20)
            # saver.log_info(f'Found {len(train_file_li)} files for training')
            saver.log_info(
                f'We have {len(kernel_names)} kernels in total: {sorted(kernel_names)}'
            )

        train_file_li = []
        val_file_li = []
        test_file_li = []
        transductive_test_file_li = []

        for gname, gname_train_file_li in sorted(train_kernel_li_map.items()):
            r1 = int(len(gname_train_file_li) * FLAGS.val_ratio_in_train_kernels)
            r2 = int(len(gname_train_file_li) * FLAGS.test_ratio_in_train_kernels)

            if FLAGS.shuffle:
                random.Random(FLAGS.random_seed).shuffle(gname_train_file_li)

            lengths = [r1, r2, len(gname_train_file_li) - r1 - r2]
            li = split_li(gname_train_file_li, lengths)
            if hasattr(FLAGS, "include_kernels") and gname not in FLAGS.include_kernels:
                continue
            if not concise:
                saver.log_info(
                    f'\tval/train: {gname} has {len(gname_train_file_li)} designs in total; split into val {len(li[0])}, '
                    f'transductive test {len(li[1])}, and train {len(li[2])}'
                )
            # assert len(li[0]) > 0
            # assert len(li[1]) > 0
            # assert len(li[2]) > 0

            val_file_li += li[0]
            transductive_test_file_li += li[1]
            train_file_li += li[2]

        for gname, val_test_file_li in sorted(val_test_kernel_li_map.items()):
            r1 = int(len(val_test_file_li) * FLAGS.val_ratio_in_test_kernels)
            if FLAGS.shuffle:
                random.Random(FLAGS.random_seed).shuffle(val_test_file_li)
            lengths = [r1, len(val_test_file_li) - r1]
            li = split_li(val_test_file_li, lengths)
            if not concise:
                saver.log_info(
                    f'\tval/test: \t{gname} has {len(val_test_file_li)} designs in total; split into val {len(li[0])} and test {len(li[1])}'
                )
            # assert len(li[0]) > 0
            # assert len(li[1]) > 0
            # assert len(li[2]) > 0
            if hasattr(FLAGS, 'test_holdout_ratio'):
                sl = shuffle_split_li_into_2_chunks(
                    li[1], ratio=FLAGS.test_holdout_ratio
                )
                li[1] = sl[
                    0
                ]  # tricky: Take the FLAGS.test_holdout_ratio amount out of li[1] and set li[1] to those
                if not concise:
                    saver.log_info(
                        f'\t\tval/test: \t{gname} due to test_holdout_ratio={FLAGS.test_holdout_ratio} further split into holdout {len(sl[0])} and others (unused) {len(sl[1])}'
                    )

            val_file_li += li[0]
            test_file_li += li[1]

        if not concise:
            saver.log_info(f'-' * 20)

        li = (train_file_li, val_file_li, test_file_li)
    else:
        assert False

    saver.log_info(
        f'{len(file_li)} graphs in total:'
        f' {len(li[0])} train ({len(li[0]) / len(file_li):.2%});'
        f' {len(li[1])} val ({len(li[1]) / len(file_li):.2%});'
        f' {len(li[2])} test ({len(li[2]) / len(file_li):.2%})'
    )
    if len(transductive_test_file_li) > 0:
        saver.log_info(
            f' {len(transductive_test_file_li)} transductive test ({len(transductive_test_file_li) / len(file_li):.2%})'
        )

    train_loader = create_dataloader(li[0], shuffle=FLAGS.shuffle, batch_size=batch_size)
    val_loader = create_dataloader(li[1], shuffle=False, batch_size=batch_size)
    test_loader = create_dataloader(li[2], shuffle=False, batch_size=batch_size)

    transductive_test_loader = None
    if len(transductive_test_file_li) > 0:
        transductive_test_loader = create_dataloader(
            transductive_test_file_li, shuffle=False, batch_size=batch_size
        )
    return train_loader, val_loader, test_loader, transductive_test_loader


def create_dataloader(
    data_li,
    shuffle,
    is_file_li=True,
    batch_size=FLAGS.batch_size,
    multi_modality=FLAGS.multi_modality,
    num_workers=FLAGS.data_loader_num_workers,
):
    if is_file_li:
        dataset = MyOwnDataset(data_files=data_li)
    else:
        dataset = data_li
    if multi_modality:
        if FLAGS.combine_fashion == 'share_final_MLPs':
            follow_batch = ['x_programl', 'x_src_code']
            if (
                hasattr(FLAGS, 'collaboration_btw_modalities')
                and FLAGS.collaboration_btw_modalities == 'edge_msgs'
            ):
                follow_batch += ['pos_node_ids_train', 'pos_node_ids']
        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            follow_batch = ['x_programl', 'x_src_code', 'x_dummy']
        else:
            raise NotImplementedError()
    else:
        if hasattr(FLAGS, 'force_data_dir'):
            # Tricky code: We assume in this case we load a multi-modality dataset!
            follow_batch = ['x_programl', 'x_src_code']
        else:
            follow_batch = None
    if FLAGS.graph_transformer_option is None:
        rtn = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=FLAGS.pin_memory,
            generator=torch_generator,
            follow_batch=follow_batch,
        )  # TODO
    else:
        rtn = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=torch_generator,
            collate_fn=collate_batch_graph_transformer,
        )  # TODO
    saver.log_info_once(
        f'Data loader created out of {len(data_li)} data points; get_num_designs: {get_num_designs(rtn)}'
    )
    return rtn


def split_li(li, lengths):
    # Taken from python 3.5 docs
    def _accumulate(iterable, fn=lambda x, y: x + y):
        'Return running totals'
        # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = fn(total, element)
            yield total

    if sum(lengths) != len(li):
        raise ValueError(
            f"Sum of input lengths does not equal the length of the input dataset! lengths={lengths}; len(li)={len(li)}"
        )

    return [
        li[offset - length : offset]
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def shuffle_split_li_into_2_chunks(li_, ratio=None, num=None, force_shuffle=False):

    if FLAGS.shuffle or force_shuffle:
        random.Random(FLAGS.random_seed).shuffle(li_)
    if ratio is not None:
        r1 = int(len(li_) * ratio)
        if r1 == 0:
            # ensure we have at least 1 data point in each of the 2 chunks...
            r1 = 1
    elif num is not None:
        r1 = num
    else:
        assert False
    lengths = [r1, len(li_) - r1]
    rtn = split_li(li_, lengths)
    return rtn


def get_kernel_samples(dataset):

    samples = defaultdict(list)
    for data in dataset:
        if FLAGS.target_kernel in data.gname:
            samples[FLAGS.target_kernel].append(data)

    return samples[FLAGS.target_kernel]


def encode_from_db(kernel_name, db_file_path):
    gexf_file = None
    for file in GEXF_FILES:
        if kernel_name in file:
            assert gexf_file is None
            gexf_file = file
    assert gexf_file is not None
    saver.log_info(f'Found {gexf_file} matching kernel name {kernel_name}')
    return get_data_list(
        gexf_files=[gexf_file],
        gexf_folder='N/A',
        db_paths_override=[db_file_path],
        to_save=False,
    )


def get_encoded_data(kernel, point, encoders, preprocessors):
    #    encoders, preprocessors = load_encoders_preprocessors()
    max_pragma_length = 17  # number obtained from get_data_list
    for gexf_file in GEXF_FILES[0:]:  # TODO: change for partial/full data
        gname = basename(gexf_file).split('.')[0]
        if kernel != gname:
            continue
        g = read_gexf_file(gexf_file)
        gname, n = _get_gname_n(gexf_file)

        xy_dict, edge_dict = encode_node_edge_dict(g, preprocessors, point)
        """ obj.point is simply parsed dict of key """
        pragmas = []
        pragma_name = []
        for name, value in sorted(point.items()):
            if type(value) is str:
                if value.lower() == 'flatten':
                    value = 2
                elif value.lower() == 'off':
                    value = 1
                elif value.lower() == '':
                    value = 3
                else:
                    raise ValueError()
            elif type(value) is int:
                pass
            else:
                raise ValueError()
            pragmas.append(value)
            pragma_name.append(name)

        if FLAGS.pragma_uniform_encoder:
            pragmas.extend([0] * (max_pragma_length - len(pragmas)))

        xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

        new_gname = gname.split('_')[0]
        if (
            FLAGS.subtask == 'train'
            and FLAGS.MAML
            and FLAGS.MAML_split_kernel
            and KEY_PRAGMA in d_node
        ):
            new_gname = gname.split('_')[0] + '-' + d_node[KEY_PRAGMA]

        d_node = defaultdict(lambda: 0)
        for k, v in xy_dict.items():
            d_node[k] = v
        d_edge = edge_dict

        data = encode_X_torch(g, d_node, d_edge, preprocessors, new_gname, "")
        return data


def get_kernel_templates(kernels):
    encoders, preprocessors = load_encoders_preprocessors()
    database = create_database()
    max_pragma_length = 17  # number obtained from get_data_list
    ret = {}
    for gexf_file in GEXF_FILES[0:]:  # TODO: change for partial/full data
        gname = basename(gexf_file).split('.')[0]
        if gname not in kernels:
            continue
        kernel = gname
        g = read_gexf_file(gexf_file)
        gname, n = _get_gname_n(gexf_file)

        db_paths = []
        for db_p in DB_PATHS:
            if FLAGS.v_db == 'v20':
                if FLAGS.only_common_db:
                    if FLAGS.test_extra:
                        paths = [
                            f
                            for f in iglob(db_p, recursive=True)
                            if f.endswith('.db')
                            and n in f
                            and 'large-size' not in f
                            and not 'archive' in f
                            and 'extra' in f
                        ]  # and not 'updated' in f
                    else:
                        paths = [
                            f
                            for f in iglob(db_p, recursive=True)
                            if f.endswith('.db')
                            and n in f
                            and 'large-size' not in f
                            and not 'archive' in f
                            and 'common-dse4' in f
                        ]  # and not 'updated' in f
                else:
                    if FLAGS.only_new_points:
                        paths = [
                            f
                            for f in iglob(db_p, recursive=True)
                            if f.endswith('extra_3.db')
                            and n in f
                            and 'large-size' not in f
                            and not 'archive' in f
                            and 'extra' in f
                        ]  # and not 'updated' in f
                    else:
                        if FLAGS.check_release_db:
                            paths = [
                                f
                                for f in iglob(db_p, recursive=True)
                                if f.endswith('.db')
                                and n in f
                                and 'large-size' not in f
                                and not 'archive' in f
                                and 'v20' in f
                                and 'v18' not in f
                                and 'test' not in f
                            ]
                        else:
                            paths = [
                                f
                                for f in iglob(db_p, recursive=True)
                                if f.endswith('.db')
                                and n in f
                                and 'large-size' not in f
                                and not 'archive' in f
                                and 'v20' in f
                                and 'v18' not in f
                                and 'one-db' in f
                                and 'test' not in f
                                and f'extended-round{FLAGS.round_num}' in f
                            ]
            elif FLAGS.v_db == 'v18':
                if FLAGS.check_release_db:
                    paths = [
                        f
                        for f in iglob(db_p, recursive=True)
                        if f.endswith('.db')
                        and n in f
                        and 'large-size' not in f
                        and not 'archive' in f
                        and 'v20' not in f
                        and 'v18' in f
                    ]
                else:
                    tag = 'one-db'
                    if FLAGS.FT_extra:
                        tag = 'extra'
                    paths = [
                        f
                        for f in iglob(db_p, recursive=True)
                        if f.endswith('.db')
                        and n in f
                        and 'large-size' not in f
                        and not 'archive' in f
                        and 'v20' not in f
                        and 'v18' in f
                        and f'{tag}-extended-round{FLAGS.round_num}' in f
                    ]  # and 'extended' not in f and 'round' not in f # and 'gae-on' in f] # and not 'updated' in f
            else:
                raise NotImplementedError
            db_paths.extend(paths)
        if db_paths is None:
            saver.warning(f'No database found for {n}. Skipping.')
            continue

        database.flushdb()
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            # print('loading', f_db)
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()
        # data = pickle.load(f_db)
        # database.hmset(0, mapping=data)
        keys = [k.decode('utf-8') for k in database.hkeys(0)]

        """ what does 'lv1' and 'lv2' in keys mean? """

        # params = keys[0].split(':')[1]
        # params = [param.split('-')[0] for param in params.split('.')]
        """param space: TILE and PARA: any number within the capability of encoder, PIPE: off, flatten, NA """
        for key in keys:
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if obj.point == {}:
                continue
            #        xy_dict, edge_dict = encode_node_edge_dict(g, preprocessors, obj.point)
            """ obj.point is simply parsed dict of key """
            pragmas = []
            pragma_name = []
            for name, value in sorted(obj.point.items()):
                if type(value) is str:
                    if value.lower() == 'flatten':
                        value = 2
                    elif value.lower() == 'off':
                        value = 1
                    elif value.lower() == '':
                        value = 3
                    else:
                        raise ValueError()
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                pragmas.append(value)
                pragma_name.append(name)

            # if FLAGS.pragma_uniform_encoder:
            #    pragmas.extend([0] * (max_pragma_length - len(pragmas)))

            # xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

            # new_gname = gname.split('_')[0]
            # if FLAGS.subtask == 'train' and FLAGS.MAML and FLAGS.MAML_split_kernel and KEY_PRAGMA in d_node:
            #    new_gname = gname.split('_')[0] + '-' + d_node[KEY_PRAGMA]

            # d_node = defaultdict(lambda : 0)
            # for k,v in xy_dict.items():
            #    d_node[k] = v
            # d_edge = edge_dict
            #
            # data = encode_X_torch(g, d_node, d_edge, preprocessors, new_gname, "")
            ret[kernel] = obj.point
    return ret


def get_data_list(
    gexf_files=GEXF_FILES,
    gexf_folder=GEXF_FOLDER,
    # db_paths_override=None,
    to_save=True,
    task=FLAGS.task,
    save_dir=SAVE_DIR,
    encoder_path=ENCODER_PATH,
):
    # base_csv = pd.read_csv(join(get_root_path(), 'dse_database', 'databases', 'base.csv'))
    # name_cycle_map = dict(zip(base_csv.Kernel_name, base_csv.CYCLE))
    saver.log_info(f'Destination save_dir={save_dir}')
    saver.log_info(f'Found {len(gexf_files)} gexf files under {gexf_folder}')
    database = create_database()

    if FLAGS.model == 'code2vec':
        preprocessors = init_preprocessors_code2vec()
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            preprocessors = init_preprocessors_multi_modality()
        else:
            if FLAGS.sequence_modeling:
                preprocessors = init_preprocessors_src_code()
            else:
                preprocessors = init_preprocessors_programl()

    data_list = []

    all_gs = OrderedDict()
    new_gs = OrderedDict()

    # X_ntype_all = []
    # X_ptype_all = []
    # X_itype_all = []
    # X_ftype_all = []
    # X_btype_all = []
    #
    # edge_ftype_all = []
    # edge_ptype_all = []
    tot_configs = 0
    num_files = 0

    init_feat_dict = {}
    max_pragma_length = 0
    if FLAGS.pragma_encoder:
        for gexf_file in tqdm(GEXF_FILES[0:]):
            saver.log_info(f'now processing {gexf_file}')
            db_paths = []
            # n = basename(gexf_file).split('_')[0]
            gname, _ = _get_gname_n(gexf_file)
            for db_p in DB_PATHS:
                paths = [
                    f
                    for f in iglob(db_p, recursive=True)
                    if f.endswith('.db')
                    and f'{gname}_result' in f
                    and 'large-size' not in f
                    and not 'archive' in f
                    and FLAGS.v_db in f
                    and f'one-db-extended-round{FLAGS.round_num}' in f
                ]  # and not 'updated' in f
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {gname}. Skipping.')
                continue
            database.flushdb()
            for idx, file in enumerate(db_paths):
                saver.log_info(f'processing db_paths for {gname}: {file}')
                with open(file, 'rb') as f_db:
                    database.hmset(0, pickle.load(f_db))
                break
            keys = [k.decode('utf-8') for k in database.hkeys(0)]
            for key in sorted(keys):
                obj = pickle.loads(database.hget(0, key))
                # try:
                if type(obj) is int or type(obj) is dict:
                    continue
                # or obj.perf == 0:#obj.ret_code.name == 'PASS':
                if task == 'regression' and key[0:3] == 'lv1':
                    continue
                if task == 'regression' and not FLAGS.invalid and obj.perf == 0:
                    continue
                # TODO !! fix databases that have this problem:
                if obj.point == {}:
                    continue
                len_pragmas = len(obj.point)
                max_pragma_length = max(max_pragma_length, len_pragmas)
                break
    else:
        max_pragma_length = 21
    saver.log_info(f'max_pragma_length={max_pragma_length}')

    all_data_df = defaultdict(list)
    debug_num_kernel_cnt = 0  # just for debugging
    # TODO: change for partial/full data
    for gexf_file in tqdm(GEXF_FILES[0:]):
        # if '2mm' in gexf_file or 'bicg' in gexf_file:
        #     continue
        # if '2mm' not in gexf_file:
        #     continue
        saver.info(f'Working on graph file: {gexf_file}')
        if FLAGS.dataset == 'vitis-cnn':
            if (
                task == 'regression'
                and FLAGS.tag == 'only-vitis'
                and 'cnn' in gexf_file
            ):
                continue
            # if 'cnn' not in gexf_file:  # TODO: change back for normal dataset
            #     continue
            pass
        elif FLAGS.dataset == 'simple-programl':
            pass
        elif FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            # proceed = False
            # for k in ALL_KERNEL:
            #     to_check = f'{k}_'
            #     if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
            #         to_check = f'{k}'
            #     if to_check in gexf_file:
            #         proceed = True
            #         break
            # if not proceed:
            #     saver.info(f'Skipping this file as the kernel name is not selected. Check config file.')
            #     continue
            pass
        else:
            raise NotImplementedError()

        # if FLAGS.dataset == 'simple-programl':
        #     gname = basename(dirname(gexf_file))
        #     # n = basename(dirname(gexf_file))
        # else:
        #     gname = basename(gexf_file).split('.')[0]
        #     # n = basename(gexf_file).split('_')[0]

        print(f'Reading gexf {gexf_file}')

        if hasattr(FLAGS, 'ignore_kernels') and len(FLAGS.ignore_kernels) > 0:
            if gname.split('_processed')[0] in FLAGS.ignore_kernels:
                saver.log_info(
                    f'Ignoring {gname} due to FLAGS.ignore_kernels={FLAGS.ignore_kernels}'
                )
                continue

        g = read_gexf_file(gexf_file)

        g.variants = OrderedDict()

        # saver.log_info(gname)

        gname, _ = _get_gname_n(gexf_file)
        saver.log_info(gname)
        all_gs[gname] = g

        # db_path = f'./all_dbs/{n}_result.db'
        if FLAGS.dataset == 'vitis-cnn':
            if n != 'cnn1':
                db_paths = glob(
                    f'{get_root_path()}/{dse_database_name}/databases/vitis/exhaustive/{gname}_result.db'
                )
                # if FLAGS.task != 'regression':
                db_paths += glob(
                    f'{get_root_path()}/{dse_database_name}/databases/vitis/bottleneck/{gname}_result.db'
                )
            else:
                db_paths = glob(
                    f'{get_root_path()}/{dse_database_name}/databases/cnn_case1/{gname}_result*.db'
                )
        elif FLAGS.dataset == 'machsuite':
            db_paths = glob(
                f'{get_root_path()}/{dse_database_name}/machsuite/databases/exhaustive/{gname}_result*.db'
            )
            db_paths += glob(
                f'{get_root_path()}/{dse_database_name}/machsuite/databases/bottleneck/{gname}_result*.db'
            )
        elif FLAGS.dataset == 'simple-programl':
            db_paths = [
                f
                for f in iglob(
                    join(
                        get_root_path(),
                        f'{dse_database_name}/simple-program/databases/**',
                    ),
                    recursive=True,
                )
                if f.endswith('.db') and gname in f and 'one-db' in f
            ]
        elif (
            FLAGS.dataset == 'programl'
            or FLAGS.dataset == 'harp'
            or FLAGS.dataset == 'harp-line-col'
        ):
            db_paths = []
            for db_p in DB_PATHS:
                # and 'extended' not in f and 'round' not in f # and 'gae-on' in f] # and not 'updated' in f
                paths = [
                    f
                    for f in iglob(db_p, recursive=True)
                    if f.endswith('.db')
                    and f'{gname}_result' in f
                    and 'large-size' not in f
                    and not 'archive' in f
                    and FLAGS.v_db in f
                    and f'one-db-extended-round{FLAGS.round_num}' in f
                ]
                db_paths.extend(paths)
            if not db_paths:
                saver.warning(
                    f'No database found for {gname} by searching in DB_PATHS {DB_PATHS}'
                )
                # continue
            # else:
            #     saver.log_info(f'Non-empty db_paths for {gname} by searching in DB_PATHS {DB_PATHS}: {db_paths}')
        elif FLAGS.dataset == 'programl-machsuite':
            # db_paths_dict = {}
            # for KERNEL in MACHSUITE_KERNEL:
            db_paths = [
                f
                for f in iglob(DB_PATHS, recursive=True)
                if f.endswith('.db')
                and n in f
                and 'large-size' not in f
                and not 'archive' in f
                and 'v20' not in f
            ]
            #    db_paths_dict[KERNEL] = db_paths
        else:
            raise NotImplementedError()

        # db_paths = sorted(db_paths)

        database.flushdb()
        saver.log_info(f'db_paths for {gname}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        # if len(db_paths) == 0:
        #     saver.log_info(f'{gname} has no db_paths')

        if len(DB_PATHS) == 0:
            raise RuntimeError(f'no database file for {gname}, db_path: {DB_PATHS}')
            # continue

        # if FLAGS.v2_db or FLAGS.FT_extra: # and (FLAGS.test_extra or FLAGS.only_new_points):
        #     if len(db_path) == 0:
        #         saver.warning(f'no database file for {n}')
        #         continue
        # else:
        #     assert len(db_paths) >= 1

        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            # print('loading', f_db)
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()
        # data = pickle.load(f_db)
        # database.hmset(0, mapping=data)
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(
            f'num keys for {gname}: {len(keys)} and lv2 keys: {len(lv2_keys)}'
        )

        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            # obj.ret_code.name == 'PASS':
            if key[0:3] == 'lv1' or obj.perf == 0:
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {gname} is {res_reference.perf}')
        else:
            saver.log_info(
                f'did not find reference point for {gname} with {len(keys)} points'
            )

        cnt = 0
        pc_links = OrderedDict()
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            # or obj.perf == 0:#obj.ret_code.name == 'PASS':
            if task == 'regression' and key[0:3] == 'lv1':
                continue
            if (
                task == 'regression'
                and not FLAGS.invalid
                and obj.perf < FLAGS.min_allowed_latency
            ):
                continue
            # TODO !! fix databases that have this problem:
            if obj.point == {}:
                # exit()
                continue
            cnt += 1
            # print(key, obj.point)
            xy_dict, edge_dict = encode_node_edge_dict(g, preprocessors, obj.point)
            if FLAGS.pc_links:
                pc_links = g.pc_links
                # print(gname, pc_links)

            # for gname_debug, g_debug in tqdm(all_gs.items()):
            #     for vname_debug, d_debug in g_debug.variants.items():
            #         d_node_debug, d_edge_debug_debug = d_debug
            #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
            #         saver.log_info_once(f'@@@@@ data 1174  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

            pragmas = []
            pragma_name = []
            for name, value in sorted(obj.point.items()):
                if type(value) is str:
                    if value.lower() == 'flatten':
                        value = 2
                    elif value.lower() == 'off':
                        value = 1
                    elif value.lower() == '':
                        value = 3
                    else:
                        raise ValueError()
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                pragmas.append(value)
                pragma_name.append(name)

            # if 'gemver' in gname:
            #     print(len(pragmas), obj.point)
            if 'gemver' in gname and len(pragmas) == 21:
                database.hdel(0, key)
                saver.warning(f'deleted {key} from database of {gname}')
                assert len(db_paths) == 1
                persist(database, db_paths[0])
                continue

            check_dim = init_feat_dict.get(gname)
            if check_dim is not None:
                # saver.info((gname, check_dim, len(pragmas)))
                assert check_dim[0] == len(pragmas), print(check_dim, len(pragmas))
                # if check_dim == len(pragmas):
                #     pass
                # else:
                #     database.hdel(0, key)
                #     print(check_dim, len(pragmas))
                #     saver.warning(f'deleted {key} from database of {gname}')
                #     assert len(db_paths) == 1
                #     persist(database, db_paths[0])
                #     continue
            else:
                init_feat_dict[gname] = [len(pragmas)]
            if FLAGS.pragma_uniform_encoder:
                pragmas.extend([0] * (max_pragma_length - len(pragmas)))

            xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

            # for gname_debug, g_debug in tqdm(all_gs.items()):
            #     for vname_debug, d_debug in g_debug.variants.items():
            #         d_node_debug, d_edge_debug_debug = d_debug
            #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
            #         saver.log_info_once(f'@@@@@ data 1233  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

            if task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(obj.perf + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = obj.perf * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert obj.perf != 0
                            # assert got_reference == True
                            y = FLAGS.normalizer / obj.perf
                            # y = res_reference.perf / obj.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y) / 2
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(
                            np.array([math.log2(res_reference.perf / obj.perf)])
                        )

                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(y + FLAGS.epsilon)
                        elif FLAGS.norm_method == 'const':
                            y = y * FLAGS.normalizer
                        elif FLAGS.norm_method == 'off':
                            pass
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname] * FLAGS.util_normalizer
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif task == 'class':
                # print(key, type(key))
                key = str(key)
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = 0 if obj.perf < FLAGS.min_allowed_latency else 1
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(
                    torch.LongTensor
                )
            else:
                raise NotImplementedError()

            vname = key

            # for gname_debug, g_debug in tqdm(all_gs.items()):
            #     for vname_debug, d_debug in g_debug.variants.items():
            #         d_node_debug, d_edge_debug_debug = d_debug
            #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
            #         saver.log_info_once(f'@@@@@ data 1296  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

            if (
                FLAGS.subtask == 'train'
                and FLAGS.MAML
                and FLAGS.MAML_split_kernel
                and KEY_PRAGMA in obj.point
            ):
                xy_dict[KEY_PRAGMA] = obj.point[KEY_PRAGMA]
            xy_dict['point'] = str(obj.point)
            g.variants[vname] = (xy_dict, edge_dict)

            # for gname_debug, g_debug in tqdm(all_gs.items()):
            #     for vname_debug, d_debug in g_debug.variants.items():
            #         d_node_debug, d_edge_debug_debug = d_debug
            #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
            #         saver.log_info_once(f'@@@@@ data 1309  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

            # if _need_fit_feature_encoder():
            #     X_ntype_all += xy_dict['X_ntype']
            #     X_ptype_all += xy_dict.get('X_ptype', [])
            #     X_itype_all += xy_dict.get('X_itype', [])
            #     X_ftype_all += xy_dict.get('X_ftype', [])
            #     X_btype_all += xy_dict.get('X_btype', [])
            #
            # edge_ftype_all += edge_dict['X_ftype']
            # edge_ptype_all += edge_dict['X_ptype']

            if (
                FLAGS.model == 'our'
                and not FLAGS.sequence_modeling
                or (FLAGS.sequence_modeling and FLAGS.data_repr == 'ast')
            ):
                X_pragma_dict_repr = xy_dict['X_pragma_dict_repr']
                df_data = {**X_pragma_dict_repr}
                for tname in TARGETS:
                    if tname in df_data:
                        df_data[tname] = xy_dict[tname].item()
                all_data_df[gname].append(df_data)
                # otherwise will have key error in data loader
                xy_dict.pop('X_pragma_dict_repr')

            debugging = False
            # debugging = True  # be careful
            if (
                FLAGS.DEBUG and cnt >= FLAGS.debug_num_data_per_kernel
            ):  # choose 7 so that when splitting into val, test, etc., there will bee >0
                saver.log_info(
                    f'Debugging! Skip other designs in this kernel; cnt={cnt}'
                )
                break  # TODO: uncomment for debugging
                # if tot_configs >= 3:
                #     print('Debugging! Skip other designs in this kernel; Only allow at most 3 designs in total')
                #     break  # TODO: uncomment for debugging
        #         print(len(pc_links), end= ' ')

        # for gname_debug, g_debug in tqdm(all_gs.items()):
        #     for vname_debug, d_debug in g_debug.variants.items():
        #         d_node_debug, d_edge_debug_debug = d_debug
        #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
        #         saver.log_info_once(f'@@@@@ data 1342  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

        if pc_links is not None:
            if len(pc_links) > 0 or gname not in new_gs.keys():
                new_gs[gname] = pc_links
        #    print(pc_links)
        #    print(new_gs[gname])
        #    xxx = input(gname)
        saver.log_info(f'final valid: {cnt}')
        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(
            f'{gname} g.variants {len(g.variants)} tot_configs {tot_configs}'
        )

        for cname, counter in preprocessors.get('counters', {}).items():
            if cname != 'num_pragmas' and cname != 'num_tokens':
                saver.log_info(f'\t{cname} {len(counter)} {counter}')

        # save all_data_df to disk
        # for gname, data_li in all_data_df.items():
        #     fn = join(saver.get_log_dir(), f'{gname.split("_")[0]}_labeled_data.csv')
        #     pd.DataFrame.from_records(data_li).to_csv(fn)
        #     saver.log_info(f'Saved csv to {fn}')

        # debugging = False
        # debugging = True # be careful
        # if debugging:
        if (
                FLAGS.DEBUG and debug_num_kernel_cnt >= FLAGS.debug_num_kernels
            ):
            # debug_num_kernel_cnt += 1
            if debug_num_kernel_cnt == 3:
                saver.log_info(f'Debugging! Skip other kernels; only need {FLAGS.debug_num_kernels} kernels in total')
                break  # TODO: uncomment for debugging
        debug_num_kernel_cnt += 1

    # itype_vocab = {}
    # if not FLAGS.sequence_modeling:
    #     enc_ntype.fit(X_ntype_all)
    #     enc_ptype.fit(X_ptype_all)
    #     if FLAGS.itype_mask_perc > 0:
    #         X_itype_all += [[MASK_TOKEN]]
    #     enc_itype.fit(X_itype_all)
    #     for itype in X_itype_all:
    #         assert len(itype) == 1
    #         itype = itype[0]
    #         if itype not in itype_vocab:
    #             itype_vocab[itype] = len(itype_vocab) + 1
    #     enc_ftype.fit(X_ftype_all)
    #     enc_btype.fit(X_btype_all)
    #
    #     if len(edge_ftype_all) != 0:
    #         enc_ftype_edge.fit(edge_ftype_all)
    #     if len(edge_ptype_all) != 0:
    #         enc_ptype_edge.fit(edge_ptype_all)

    # for gname_debug, g_debug in tqdm(all_gs.items()):
    #     for vname_debug, d_debug in g_debug.variants.items():
    #         d_node_debug, d_edge_debug_debug = d_debug
    #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
    #         saver.log_info_once(f'@@@@@ data 1365  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        for gname, pc in new_gs.items():
            all_gs[gname].pc_links = pc
    #    print(pc)

    # for gname, g in all_gs.items():
    #    print(g.pc_links)
    # xxx = input(gname)

    scalers = {}
    if FLAGS.target_preproc != 'None':
        saver.log_info('minmax_scaling...')
        ys_dict = defaultdict(list)
        for gname, g in tqdm(all_gs.items()):
            for vname, (d_node, _, _) in g.variants.items():
                for tname in TARGETS:
                    y = d_node[tname].item()
                    ys_dict[tname].append(y)
        ys_dict_trans = {}
        for tname in TARGETS:
            if FLAGS.target_preproc == 'minmax':
                scaler = MinMaxScaler()
            elif FLAGS.target_preproc == 'z-score':
                scaler = StandardScaler()
            else:
                raise ValueError()
            y_li = np.array(ys_dict[tname]).reshape(-1, 1)
            scaler.fit(y_li)
            if FLAGS.target_preproc == 'minmax':
                saver.log_info(
                    f'scalar for {tname} range: {scaler.data_min_} {scaler.data_max_} '
                    f'with {scaler.n_samples_seen_} examples'
                )
            elif FLAGS.target_preproc == 'z-score':
                saver.log_info(
                    f'scalar for {tname} mean: {scaler.mean_} std {scaler.var_} '
                    f'with {scaler.n_samples_seen_} examples'
                )
            else:
                raise ValueError()
            ys_dict_trans[tname] = scaler.transform(y_li)
            scalers[tname] = scaler
        i = 0
        for gname, (g, _) in tqdm(all_gs.items()):
            for vname, (d_node, _, _) in g.variants.items():
                for tname in TARGETS:
                    d_node[tname] = torch.FloatTensor(
                        np.array([ys_dict_trans[tname][i]])
                    )
                i += 1
        saver.log_info('minmax_scaling done')

    saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')
    # saver.log_info(f'GEXF_FILES={GEXF_FILES} (len={len(GEXF_FILES)})')

    # for gname_debug, g_debug in tqdm(all_gs.items()):
    #     for vname_debug, d_debug in g_debug.variants.items():
    #         d_node_debug, d_edge_debug_debug = d_debug
    #         g_new_debug = d_node_debug['n_dict_src_code']['g_new']
    #         saver.log_info_once(f'@@@@@ data 1425  {gname_debug} has nodes: {g_new_debug.number_of_nodes()}')

    seen_pragma_trans_types = set()
    # if not FLAGS.sequence_modeling:

    if FLAGS.model == 'code2vec':
        fit_preprocessors_code2vec(preprocessors)
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            fit_preprocessors_multi_modality(preprocessors)
        else:
            if FLAGS.sequence_modeling:
                fit_preprocessors_src_code(preprocessors)
            else:
                fit_preprocessors_programl(preprocessors)

    saver.log_info(f'Sklearn encoders fitted')
    # for cname, counter in preprocessors.get('counters', {}).items():
    # saver.log_info(f'\t{cname} {len(counter)} {counter}')

    # if FLAGS.ptrans:  # TODO: handle pragma "nodes" for sequence modeling
    #     # Fill in empty tensor for pragma transformation nids.
    #     for gname, g in tqdm(all_gs.items()):
    #         for vname, d in g.variants.items():
    #             d_node, d_edge = d
    #             seen_pragma_trans_types.update(d_node['X_pragma_trans'].keys())
    #     saver.log_info(
    #         f'Found {len(seen_pragma_trans_types)} pragma transformation module types: {seen_pragma_trans_types}')
    #     for gname, g in tqdm(all_gs.items()):
    #         num_nodes = g.number_of_nodes()
    #         for vname, d in g.variants.items():
    #             d_node, d_edge = d
    #             for tstype in seen_pragma_trans_types:
    #                 cur_nids = d_node['X_pragma_trans'].get(tstype)
    #                 new_nids = torch.zeros(num_nodes)
    #                 if cur_nids is not None:
    #                     new_nids[cur_nids] = 1
    #                 d_node['X_pragma_trans'][tstype] = new_nids

    # for gname, g in tqdm(all_gs.items()):
    #     for vname, d in g.variants.items():
    #         d_node, d_edge = d
    #         g_new = d_node['n_dict_src_code']['g_new']
    # saver.log_info_once(f'@@@@@ data 1470  {gname} has nodes: {g_new.number_of_nodes()}')

    saver.log_info('Start encoding gs')

    for gname, g in tqdm(all_gs.items()):
        # edge_index = create_edge_index_both(g)
        saver.log_info('edge_index created', gname)
        new_gname = gname.split('_')[0]

        # saver.log_info(f'@@@@ 1481 {gname} --> {new_gname}')

        if (
            FLAGS.subtask == 'train'
            and FLAGS.MAML
            and FLAGS.MAML_split_kernel
            and KEY_PRAGMA in d_node
        ):
            new_gname = gname.split('_')[0] + '-' + d_node[KEY_PRAGMA]

        for vname, d in g.variants.items():
            d_node, d_edge = d

            # g_new = d_node['n_dict_src_code']['g_new']
            # saver.log_info_once(f'@@@@@ data 1492  gname={gname} new_gname={new_gname} has nodes: {g_new.number_of_nodes()}')

            data = encode_X_torch(
                g, d_node, d_edge, preprocessors, new_gname, vname, task=task
            )

            if task == 'regression':
                data_list.append(data)
            elif task == 'class':
                data_list.append(data)
            else:
                raise NotImplementedError()

    if FLAGS.model == 'code2vec':
        from data_code2vec import not_found_cnt

        saver.log_info(
            f'Done loading code2vec data (i.e. pretrained design embeddings); not_found_cnt={not_found_cnt}'
        )
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
                from data_multi_modality import PC_LINKS_DICT

                save_pickle(
                    PC_LINKS_DICT, join(saver.get_obj_dir(), 'pc_links_dict.pkl')
                )

                print_stats(
                    preprocessors['preprocessors_src_code']['counters'][
                        'num_matched_pairs'
                    ],
                    'number of matched node-token pairs per program',
                )
        else:
            if FLAGS.sequence_modeling:
                nns = [len(d.x) for d in data_list]
                print_stats(
                    nns,
                    'number of chunks for pure seq or number of nodes for ast',
                    saver,
                )
                from data_src_code import FUNCTION_NAMES, VARIABLE_NAMES

                saver.log_info(
                    f'FUNCTION_NAMES={FUNCTION_NAMES}'
                )  # so that we can enrich the tokenizer with these additional names
                saver.log_info(f'VARIABLE_NAMES={VARIABLE_NAMES}')

                print_stats(
                    preprocessors['counters']['num_tokens'],
                    'number of tokens per program',
                )

            else:
                nns = [d.x.shape[0] for d in data_list]
                print_stats(nns, 'number of nodes')

                from data_programl import diameters_list

                print_stats(diameters_list, 'graph diameter')

        # exit()

        if not FLAGS.sequence_modeling:
            if not FLAGS.only_pragma_nodes:
                nnodes = [d.x.shape[0] for d in data_list]
                print_stats(nnodes, 'num nodes')

                nedges = [d.edge_index.shape[1] for d in data_list]
                print_stats(nedges, 'num edges')

                ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
                print_stats(ads, 'avg degrees')

                print_stats(
                    preprocessors['counters']['num_pragmas'],
                    'number of pragmas per program',
                )

    # for d in data_list:
    #     print(d.num_features)

    # ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    # print_stats(ads, 'avg degrees')
    if len(data_list) == 0:
        raise RuntimeError(
            f'len(data_list) == 0! Check db files/paths by opening log.txt'
        )
    saver.info(data_list[0])

    saver.log_info(f'dataset[0].num_features {data_list[0].num_features}')
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        # if target == 'quality':
        #     continue
        plot_dist(
            ys,
            f'{target}_ys',
            saver.get_log_dir(),
            saver=saver,
            analyze_dist=True,
            bins=None,
        )
        if FLAGS.task == 'class':
            saver.log_info(f'{target}_ys: {Counter(ys)}')

    # from sequence_modeling import CODE_LEN_LI
    # print_stats(CODE_LEN_LI, 'CODE_LEN_LI')

    timer = OurTimer()

    if FLAGS.force_regen and to_save:
        saver.log_info(
            f'Saving {len(data_list)} to disk {save_dir}; Deleting existing files'
        )
        try:
            rmtree(save_dir)
        except Exception as e:
            saver.log_info(e)
        create_dir_if_not_exists(save_dir)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(save_dir, 'data_{}.pt'.format(i)))
        saver.log_info(f'save data_list {timer.time_and_clear()}')

    if FLAGS.force_regen and to_save:
        # if FLAGS.only_pragma:
        #     obj = {'enc_ptype': enc_ptype}
        # else:
        # obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype,
        #        'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
        #        'enc_btype': enc_btype,
        #        'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge,
        #        'itype_vocab': itype_vocab
        #        }

        # if FLAGS.model == 'code2vec':
        #     saver.log_info(f'code2vec: Skip encoder saving)')
        # else:
        # assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            save(
                {
                    'encoders_programl': preprocessors['preprocessors_programl'][
                        'encoders'
                    ],
                    'encoders_src_code': preprocessors['preprocessors_src_code'][
                        'encoders'
                    ],
                },
                encoder_path,
            )
        else:
            # if FLAGS.encoder_path == None:
            save(preprocessors['encoders'], encoder_path)
        saver.log_info(f'save encoders {timer.time_and_clear()}')

        save(preprocessors, get_preprocessors_path(encoder_path))
        saver.log_info(f'save preprocessors {timer.time_and_clear()}')

        if FLAGS.pragma_uniform_encoder:
            for gname in init_feat_dict:
                init_feat_dict[gname].append(max_pragma_length)
        name = 'pragma_dim'
        # if FLAGS.v2_db:
        #     save(obj_p, join(save_dir, name))
        #     name += '_v2'
        save(init_feat_dict, join(save_dir, name))
        saver.log_info(f'save init_feat_dict {timer.time_and_clear()}')

        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim[0]}')

    if not to_save:
        return data_list

    rtn = MyOwnDataset(init_feat_dict=init_feat_dict, save_dir=save_dir, encoder_path=encoder_path)

    rtn.add_target_scalers(scalers, to_save=to_save)
    rtn.add_attribute('seen_pragma_trans_types', sorted(list(seen_pragma_trans_types)))
    # rtn.add_attribute('init_feat_dict', init_feat_dict)

    return rtn, init_feat_dict


def read_gexf_file(gexf_file, multi_modality=None, sequence_modeling=None):
    if multi_modality is None:
        multi_modality = FLAGS.multi_modality
    if sequence_modeling is None:
        sequence_modeling = FLAGS.sequence_modeling
    if FLAGS.model == 'code2vec':
        g = read_code2vec_data(gexf_file)
    else:
        assert FLAGS.model == 'our'
        if multi_modality:
            g = read_multi_modality_data(gexf_file)
        else:
            if sequence_modeling:
                g = read_source_code(gexf_file)
                # e.g. aes kernel's ast geenration is currently not working (manually obtained from https://github.com/yunshengb/software-gnn/blob/atefeh/dse_database/machsuite/dot-files/original-size/aes_kernel.c.dot)
                if g is None:
                    raise ValueError(f'Cannot load the graph for {gexf_file}; skip')
                    # continue
            else:
                g = read_programl_graph(gexf_file)
    return g


def encode_node_edge_dict(
    g, preprocessors, point, multi_modality=None, sequence_modeling=None
):
    if multi_modality is None:
        multi_modality = FLAGS.multi_modality
    if sequence_modeling is None:
        sequence_modeling = FLAGS.sequence_modeling
    if FLAGS.model == 'code2vec':
        xy_dict, edge_dict = encode_feat_dict_code2vec(g, preprocessors, point=point)
    else:
        assert FLAGS.model == 'our'
        if multi_modality:
            xy_dict, edge_dict = encode_feat_dict_multi_modality(
                g, preprocessors, point=point
            )
        else:
            if sequence_modeling:
                xy_dict, edge_dict = encode_feat_dict_src_code(
                    g, preprocessors, point=point
                )
            else:
                xy_dict, edge_dict = encode_feat_dict_programl(
                    g, preprocessors, point=point
                )
    return xy_dict, edge_dict


def encode_X_torch(
    g,
    d_node,
    d_edge,
    preprocessors,
    new_gname,
    vname,
    task=FLAGS.task,
    multi_modality=None,
    sequence_modeling=None,
    cached_edge_index=None,
    cached_edge_attr=None,
):
    if multi_modality is None:
        multi_modality = FLAGS.multi_modality
    if sequence_modeling is None:
        sequence_modeling = FLAGS.sequence_modeling
    if FLAGS.model == 'code2vec':
        data = encode_X_torch_code2vec(
            g, d_node, d_edge, preprocessors, new_gname, vname
        )
    else:
        assert FLAGS.model == 'our'
        if multi_modality:
            data = encode_X_torch_multi_modality(
                g, d_node, d_edge, preprocessors, new_gname, vname, task=task, cached_edge_index=cached_edge_index, cached_edge_attr=cached_edge_attr
            )
        else:
            if sequence_modeling:
                # saver.log_info(f'sequence_modeling={sequence_modeling}')
                data = encode_X_torch_src_code(
                    d_node, d_edge, preprocessors, new_gname, vname, task=task
                )
            else:
                data = encode_X_torch_programl(
                    g, d_node, d_edge, preprocessors, new_gname, vname, task=task, cached_edge_index=cached_edge_index, cached_edge_attr=cached_edge_attr
                )
    return data


def _get_gname_n(gexf_file):

    if FLAGS.dataset == 'simple-programl':
        # gname = basename(dirname(gexf_file))
        # n = basename(dirname(gexf_file))
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}_"

    elif FLAGS.dataset == 'harp' or FLAGS.dataset == 'harp-line-col':
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}_"

    else:
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}_"

    # if 'stencil_stencil2d' in gname:
    #     exit()
    gname = gname.replace('stencil_stencil2d', 'stencil')
    gname = gname.replace('_processed_result', '')
    assert 'processed' not in gname

    return gname, n


# def _need_fit_feature_encoder():
#     if FLAGS.sequence_modeling:
#         if FLAGS.data_repr == 'ast':
#             return True
#         else:
#             return False  # pure sequence
#     else:
#         return True


def create_edge_index_both(g_programl):
    edge_index = create_edge_index(g_programl)

    return edge_index


def _get_y(data, target):
    return getattr(data, target.replace('-', '_'))


def print_data_stats(data_loader, tvt):
    nns, ads, ys = [], [], []
    for d in tqdm(data_loader):
        nns.append(d.x.shape[0])
        # ads.append(d.edge_index.shape[1] / d.x.shape[0])
        ys.append(d.y.item())
    print_stats(nns, f'{tvt} number of nodes')
    # print_stats(ads, f'{tvt} avg degrees')
    plot_dist(
        ys, f'{tvt} ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None
    )
    saver.log_info(f'{tvt} ys', Counter(ys))


def load_encoders_preprocessors(save_dir=SAVE_DIR, encoder_path=ENCODER_PATH):
    path = encoder_path
    if path == 'None' or path is None:
        # saver.log_info(f'encoder path is None; use default path instead')
        path = ENCODER_PATH
    rtn = load(path, print_msg=False)
    # saver.log_info(f'Loaded encoder from {path}')
    if rtn is None:
        raise ValueError(
            f'Trying to load encoders; Check encoder path {path}; Maybe force regen '
            f'or check load_encoders path'
        )
    preprocessors_path = join(save_dir, 'preprocessors.klepto')
    preprocessors = load(preprocessors_path, print_msg=False)
    # saver.log_info(f'Loaded encoder from {path}')
    if preprocessors is None:

        raise ValueError(
            f'Trying to load preprocessors; Check preprocessors_path {preprocessors_path}'
        )
    return rtn, preprocessors


def get_preprocessors_path(encoder_path):
    rtn = join(dirname(encoder_path), 'preprocessors.klepto')
    # saver.log_info(f'get_preprocessors_path(): Inferring path from {encoder_path} as {rtn}')
    return rtn


def _encode_fulltext(g, text_encoder):
    X_fulltext = []

    for node, ndata in sorted(g.nodes(data=True)):  # TODO: node ordering
        with torch.no_grad():
            s = ndata.get('full_text', '')
            if s == '':
                s = ndata.get('content', '')
            X = text_encoder.encode(s)
            # Note: X_fulltext i trainable
            assert X.shape == (1, text_encoder.dim())

        X_fulltext.append(X)

    return torch.cat(X_fulltext, dim=0)


def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype):
    """
    edge_dict is the dictionary returned by _encode_edge_dict
    """
    X_ftype = enc_ftype.transform(edge_dict['X_ftype'])
    X_ptype = enc_ptype.transform(edge_dict['X_ptype'])

    X = hstack((X_ftype, X_ptype))
    X = coo_to_sparse(X, FLAGS.device)
    X = X.to_dense()

    return X


# def get_num_features(dataset, sequence_modeling=FLAGS.sequence_modeling):
# Above code is buggy. This is a tricky bug. When FLGGS is updated,
# sequence_modeling should be also updated, but somehow sequence_modeling remains
# as the old value in FLAGS.
# Therefore, should use the following code.
def get_num_features(dataset, sequence_modeling=None):
    if sequence_modeling is None:
        sequence_modeling = FLAGS.sequence_modeling
    if sequence_modeling:
        from data_src_code import get_code_encoder_dim

        num_features = get_code_encoder_dim()
        if FLAGS.data_repr == 'ast' and FLAGS.AST_combine_node_edge_labels:
            return num_features + dataset[0].X_ast_node_labels.shape[1]
    else:
        num_features = dataset[0].num_features
        if num_features == 0:
            num_features = dataset[0].x_programl.shape[1]
    return num_features


def get_edge_dim(dataset):
    if hasattr(dataset[0], 'edge_attr'):
        edge_attr = dataset[0].edge_attr
        assert edge_attr is not None, f'Check sequence_modeling; should be False...?'
        edge_dim = edge_attr.shape[1]
    elif hasattr(dataset[0], 'edge_attr_programl'):
        edge_dim = dataset[0].edge_attr_programl.shape[1]
    else:
        assert False
    return edge_dim


def get_num_designs(data_loader):
    # assert isinstance(data_loader, PyGDataLoader)
    # len(data_loader) would be the number of batches
    # saver.log_info(f'len(data_loader)={len(data_loader)}; len(data_loader.dataset)={len(data_loader.dataset)}')
    if hasattr(data_loader.dataset, 'get_num_designs'):
        return data_loader.dataset.get_num_designs()  # PairwiseDataloader
    else:
        return len(data_loader.dataset)
