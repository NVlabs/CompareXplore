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


from types import SimpleNamespace
from pathlib import Path
import torch
from utils import get_user, get_host
from utils import check_prepend_root_folder, get_best_gpu
from collections import OrderedDict


model = 'our'
# model = 'tfdse'
# model = 'code2vec'
# model = 'OpenAI'
# model = 'pytriton'


# DEBUG = True  # will generate less data for faster local debugging
DEBUG = False

disable_src_code = False

dse_start_idx = 0
dse_end_idx = 42
disable_gnn = False

# HARP_setup = False
HARP_setup = True  

# critical: choose which dataset
# v2_db = True  # v20; 4k designs
# v2_db = False  # v18; 04/14/2023: 8k designs; (earlier) 7k designs

# task = 'class'
task = 'regression'
# task = 'rl'


# subtask = 'train'
subtask = 'inference'
# subtask = 'inference_dse'
# subtask = 'dse'
# subtask = 'visualize'
# subtask = 'analyze_data'

# parser.add_argument('--v2_db', default=v2_db)  # if set to true uses the db of the new version of the tool: 2020.2
v_db = 'v21'

# v_db = 'v20'
# v_db = 'v18'

if v_db != 'v18':
    only_common_db = False
    test_extra = False
    only_new_points = False

round_num = 1 if v_db == 'v21' else 3 if v_db == 'v20' else 13


load_model = 'None'


load_keys = 'None'
load_db = 'None'
load_db_kernel = 'None'
run_kernel = 'None'


if load_model != 'None':
    load_model_weights = True
    # load_model_weights = False  # be careful! just load the architecture but still randomly initialized


mode = 'standalone'  # do not change it here
# another mode (acc_launch) is set by launch.py


sequence_modeling = False  # programl only
# sequence_modeling = True  # LM only or GNN+LM
if model != 'our':
    sequence_modeling = False
if model in ['OpenAI', 'pytriton']:
    sequence_modeling = (
        True  # to enable some data generation steps such as SOURCE_FILES init
    )
    if model == 'OpenAI':
        # OpenAI_model = 'gpt-3.5-turbo'
        OpenAI_model = 'gpt-4'

replace_with_random_weights = None
combine_node_edge_labels = None
data_repr = None
code_encoder = 'codet5'  # default so that force_data_dir will work
chunk_offset = None


multi_modality = False
# multi_modality = True

combine_fashion = None
feed_p_to_tf = None
interleave_GNN_transformer = False
if model != 'our':
    multi_modality = False

if model == 'code2vec':
    code2vec_data_version = 'neurips2023_main'
    # code2vec_data_version = 'neurips2023_benchmark'


# use_redis = True  # default
use_redis = False

if sequence_modeling:
    # data_repr = 'full'
    # data_repr = 'simplified'
    data_repr = 'penhance'  # pragma-enhanced
    # data_repr = 'ast'

    # code_encoder = 'codebert'
    # code_encoder = 'graphcodebert'
    code_encoder = 'codet5'  # codet5-small
    # code_encoder = 'TheBloke/CodeLlama-7B-GPTQ'
    # code_encoder = 'codellama/CodeLlama-7b-hf'
    # code_encoder = 'codet5-large'
    # code_encoder = 'codegen-350M-multi'
    # code_encoder = 'OpenAI'  # linear probing

    if code_encoder == 'OpenAI':
        OpenAI_embedding_model = 'text-embedding-3-large'  # dim=3072
        save_cache_every_new_io_pairs = 1  # very frequent saving,,,

    use_peft = False
    # use_peft = True

    if use_peft:
        peft_r = 16  # dimension of the updated matrices
        lora_alpha = 16  # parameter for scaling

        # peft_r=2  # dimension of the updated matrices
        # lora_alpha=2  # parameter for scaling

        lora_dropout = 0.1

    chunk_emb = 'cls'
    # chunk_emb = 'pooler'

    finetune_bert = True
    # finetune_bert = False # save memory

    replace_with_random_weights = False  # default; esp for testing
    # replace_with_random_weights = True  # Be careful! This is to replace pretrained model with random model!

    # use default MAX_LEN by the tokenizer (256 for codet5)
    max_code_tokens_len = None
    # max_code_tokens_len = 128
    # max_code_tokens_len = 64
    # max_code_tokens_len = 32
    # max_code_tokens_len = 16
    # if not v2_db:
    #     max_code_tokens_len = 512
    # max_code_tokens_len = 1024
    # max_code_tokens_len = 2048

    # be efficient!
    max_code_tokens_len = 64
    # max_code_tokens_len = 128
    # max_code_tokens_len = 256

    if data_repr == 'ast':
        # AST_combine_node_edge_labels = False
        AST_combine_node_edge_labels = True

        # max_code_tokens_len = 64
        max_code_tokens_len = 32

        bi_directional_AST = False
        # bi_directional_AST = True # model both directions in AST and handle edge labels using GNN

    else:  # pure seq
        # add_edges_fc_graph = False
        add_edges_fc_graph = True

        # chunk_offset = 0 # default; moving window moves without any offset --> non-overlapping
        # chunk_offset = 192  # overlapping chunks
        # chunk_offset = 500 # overlapping chunks

        # be efficient!
        chunk_offset = 16
        # chunk_offset = 32
        # chunk_offset = 64

    # token_att_masking = False
    token_att_masking = True

    # preserve_keywords = False
    preserve_keywords = True

    if preserve_keywords:
        # pk_version = 1
        pk_version = 2

    if code_encoder in ['codet5', 'codet5-large']:
        bypass_tf_layers = False  # default
        # bypass_tf_layers = True # crazy: bypass some codet5 encoder layers

        if bypass_tf_layers:
            # keep_layers = [0]
            # keep_layers = [5]
            keep_layers = [0, 1, 2]
            # keep_layers = [] # crazy; no token-token attention at all; no encoder block used at all!

    # apply_act_conv_first = True # default
    apply_act_conv_first = False

    vis_transformer_att = False  # default
    # vis_transformer_att = True # will run vis and exit()


force_regen = False
# force_regen = True


if not force_regen:
    if DEBUG:
        # DEBUG
        force_data_dir = check_prepend_root_folder(
            f'../../save/programl/v21_True-1-epbch-{task}_False_False_speedup-log2_False_whole-machsuite-poly_DEBUG_programl_False_False_None_None_nosplit_regular_encoder_True_s_penhance_{code_encoder}_64_tm_pk_v2_fc_co16_programl+src_code_feed'
        )
    else:
        force_data_dir = check_prepend_root_folder(
            f'../../save/programl/v21_True-1-epbch-{task}_False_False_speedup-log2_False_whole-machsuite-poly_programl_False_False_None_None_nosplit_regular_encoder_True_s_penhance_{code_encoder}_64_tm_pk_v2_fc_co16_programl+src_code_feed'
        )

    # Turn off fancy stuff above!
    force_data_dir = (
        None  # turn off force_data_dir; just load whatever should be loaded
    )

else:
    if DEBUG:
        debug_num_data_per_kernel = 50
        debug_num_kernels = 3

full_quadratic_attention = False
pc_links = None
if multi_modality:
    what_modalities = 'programl+src_code'

    if what_modalities == 'programl+src_code':
        assert sequence_modeling and data_repr == 'penhance'
    else:
        raise NotImplementedError()

    combine_fashion = 'share_final_MLPs'
    # combine_fashion = 'share_GNNs_MLPs'

    if combine_fashion == 'share_GNNs_MLPs':
        # add_pragma_links = False  # default
        add_pragma_links = True  #

        multi_glevel_embs = False  # default
        # multi_glevel_embs = True

    # elif combine_fashion == 'p_to_tf':
    #     feed_what = 'summary'
    #     parser.add_argument('--feed_what', type=str, default=feed_what)

    if combine_fashion == 'share_final_MLPs':
        freeze_code_encoder = False

        # feed_p_to_tf = False  # default
        feed_p_to_tf = True
        if disable_src_code == True:
            feed_p_to_tf = False

        if feed_p_to_tf:
            # which_pos_to_take = '0' # 0 indicates the programl (CDFG) summary embedding
            # which_pos_to_take = '1' # 1 indicates the <s> embedding
            which_pos_to_take = '0_and_1'  # both
            node_token_interact_start_layer = 1
            # two_modalities_interact_start_layer = -1
            two_modalities_interact_start_layer = 1

        # pc_links = False  # default
        pc_links = True
        if disable_src_code == True:
            pc_links = False

        if pc_links:
            alignment_decoder = 'concat_MLP'  # default
            # alignment_decoder = 'dot'
            # alignment_decoder = 'cosine'

            # pc_links_aug = None  # only the specific token indicated by <line> <col>
            # pc_links_aug = 'pseudo' # block node token interaction
            # pc_links_aug = None  # only the specific token indicated by <line> <col>
            pc_links_aug = 'pseudo'  # block node token interaction
            # pc_links_aug = 'all_line' # all tokens on the line are matched
            # pc_links_aug = 'all_line_sw'  # strong weak distinction
            # pc_links_aug = 'all_line_swp'  # strong weak distinction + add pragma links with type 'p' with 'GreaseLM'
            # pc_links_aug = 'all_line_swp_grease'  # strong weak distinction + add pragma links with type 'p'
            # pc_links_aug = 'grease'  # 'GreaseLM' baseline
            # pc_links_aug = 's_grease'

            node_token_interaction = True

            pc_links_holdout_ratio = 0  # default
            if pc_links_aug is not None and node_token_interaction:
                # pc_links_holdout_ratio = 0.4
                pc_links_holdout_ratio = 0  # critical: need to be tuned
                if pc_links_aug == 'grease':
                    pc_links_holdout_ratio = 0

            if 'inference' in subtask:
                pc_links_force_use_train_when_inference = False
                # pc_links_force_use_train_when_inference = True # be careful! This will use partial pc links

            if node_token_interaction:
                # project_node_embs_before_align = False
                project_node_embs_before_align = True

                full_quadratic_attention = False  # default
                # full_quadratic_attention = True # bypass message passing via bridge links -- instead performs full att

                actually_aggregate = True  # default
                # actually_aggregate = False

                collaboration_btw_modalities = None
                # collaboration_btw_modalities = 'edge_msgs'

                # If inter-levae GNN and transformer layer by layer,
                # need to (1) do cross-modality exchange in both directions, and
                # (2) if further feed_p_to_tf, need to prepare the token ids by taking that fed CDFG embedding into
                # consideration.
                # interleave_GNN_transformer = False  # default
                interleave_GNN_transformer = True  # much fancier

                if interleave_GNN_transformer:
                    weight_tying_for_interaction_GNN = False
                    # weight_tying_for_interaction_GNN = True

                    # just interact one more time (like an 'encore') after all layers
                    interact_after_all_layers = True
                    # interact_after_all_layers = False # to be safe, no more interaction at the end

                    # otherwise some nodes with bridge information may receive information iteratively, causing large embedding values
                    apply_norm_after_interaction = True
                    # apply_norm_after_interaction = False

                    if apply_norm_after_interaction:
                        emb_norm_method = 'L2'
                        # emb_norm_method = 'graph'

                    apply_mlp_after_interaction = True
                    # apply_mlp_after_interaction = False

            # default; node-token level (fine-grained)
            node_token_alignment_loss = False
            # node_token_alignment_loss = True

            gs_alignment_loss = False  # default; graph-sequence level (coarse)
            # gs_alignment_loss = True


if load_db != 'None':
    load_db = check_prepend_root_folder(load_db)


if load_model != 'None':
    load_model = check_prepend_root_folder(load_model)
    if '.pth' not in load_model and 'pt' not in load_model:
        load_model += '/val_model_state_dict.pth'

# test_kernel = 'gesummv'


# dataset = 'vitis-cnn'

# dataset = 'machsuite'
# dataset = 'programl-machsuite'

dataset = 'programl'  # machsuite and poly in GNN-DSE

# dataset = 'programl'  # machsuite and poly in GNN-DSE
# dataset = 'harp'
# dataset = 'harp-line-col'

# dataset = 'simple-programl'


benchmarks = ['machsuite', 'poly']
if dataset == 'simple-programl':
    benchmarks = ['simple']


# tag = 'only-vitis'
# tag = 'whole-machsuite'


tag = 'whole-machsuite-poly'  
if dataset == 'simple-programl':
    tag = 'simple'
# tag = 'gemm-blocked'


# check_release_db = False  # default
check_release_db = False  # for released version of the folder `dse_database`


dse_database_name = 'dse_database'  # default
if check_release_db:
    dse_database_name = 'dse_database_06122023_v2'


decoder_arch = []
####
# updated-1 --> fine-tuned-dse1
# updated-2 --> common
# updated-3 --> fine-tune-dse2
# updated-4 --> fine-tune-dse3
# updated-5 --> fine-tune-dse4
# updated-old-5 --> fine-tune-oldspeed-fromv1-todse4
# updated-new-5 --> fine-tune-todse4-fromv1
# updated-new-6 --> fine-tune-todse5-fromv1
# updated-new-5-norm-util --> fine-tune-todse5-fromv1-norm-util
# updated-yizhou-5 --> fine-tune-dse4-fromv1-twosetp-yizhou
# updated-new-6 --> fine-tune-todse5-fromv1 with results of new-5
# updated-freeze4-5 --> fine-tune-todse4-fromv1-freeze4
# updated-freeze5-5 --> fine-tune-todse4-fromv1-freeze5
# updated-onlynew-tuneall-5 --> fine-tune-dse4-only-new-points-tune-all
####

TARGETS = [
    'perf',
    'quality',
    'util-BRAM',
    'util-DSP',
    'util-LUT',
    'util-FF',
    'total-BRAM',
    'total-DSP',
    'total-LUT',
    'total-FF',
]

# MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
#                     'nw', 'md', 'stencil-3d']


# graph_type = ''  # original DAC22 graph
# graph_type = 'extended-pseudo-block-base' ## check that the connected ones are not used
# graph_type = 'extended-pseudo-block-connected'
graph_type = 'extended-pseudo-block-connected-hierarchy'
# graph_type = ''
# if sequence_modeling:
#     graph_type = ''


graph_transformer_option = None
# graph_transformer_option = True
if graph_transformer_option:
    graph_transformer_option = {
        'graph_encoding': ['degree', 'Laplacian'],
        'Laplacian_K': 20,
        'is_undirected': False,
    }
    # conv_type = 'mha'
    conv_type = 'gps_conv'  # slow and memory-heavy
    graph_transformer_option['conv_type'] = conv_type
    if conv_type == 'mha':
        graph_transformer_option['num_heads'] = 1
        graph_transformer_option['attention_map_aug'] = 'proximity'
    elif conv_type == 'gps_conv':
        graph_transformer_option['num_heads'] = 2
        graph_transformer_option['need_local_mpnn'] = True
    else:
        assert False
if sequence_modeling:
    graph_transformer_option = None


# graph auto encoder
# gae_T = True
gae_T = False
if sequence_modeling:
    gae_T = False
gae_T = False


gae_P = False

if gae_P:
    input_encode = False

    # decoder_type = 'None'
    # decoder_type = 'type1'
    decoder_type = 'type1'


# self-supervised learning
SSL = False


# load_pretrained_GNN = False  # default

load_pretrained_GNN = True
if subtask == 'train':
    load_pretrained_GNN = True

load_pretrained_GNN = False
# if sequence_modeling and not multi_modality:
#     load_pretrained_GNN = False
if task == 'class':
    load_pretrained_GNN = False
if graph_transformer_option is not None:
    # may have additional node positional encoding (shape mismatch)
    load_pretrained_GNN = False


if load_pretrained_GNN:
    pretrained_GNN_name = 'our'
    # pretrained_GNN_name = 'GraphMAE'

    guide_loss_w = 0.1
    guide_loss_w = 0.003  # TODO: which to use????


load_guidance_emb = False
if load_guidance_emb:
    guidance_emb_path = 'pretrain_gnn/zongyue_code/pretrained_guide_emb.pth'


class_model_path = None


target_kernel = None
# # target_kernel = 'gemm-blocked'
# parser.add_argument('--target_kernel', default=target_kernel)
# # kernels = ['jacobi-1d', '3mm', 'fdtd-2d', 'gemm-p', 'gemver']
# # kernels = ['gemver']
# # kernels = ['gemm-p']

# if target_kernel == None:
all_kernels = True
# else:
#     all_kernels = False

sample_finetune = False


FT_extra = False


new_speedup = True  # new_speedup: same reference point across all,


# old_speedup: base is the longest latency and different per kernel


# if set to true GNN part will be fixed and only MLP will be trained
feature_extract = False


if feature_extract:
    random_MLP = False  # true: initialize MLP randomly

    fix_gnn_layer = None  # if none, all layers will be fixed
    fix_gnn_layer = (
        5  # number of gnn layers to freeze, feature_extract should be set to True
    )
    # if not set to none, feature_extract should be True


# test_kernels = None

test_kernels = ['gemver', 'fdtd-2d', 'correlation', 'adi', 'trmm-opt', '3mm']
if HARP_setup:
    test_kernels = []


if 'dse' in subtask:
    if not HARP_setup:
        # choose_test_kernel = 0
        # choose_test_kernel = 1
        # choose_test_kernel = 2
        # choose_test_kernel = 3
        # choose_test_kernel = 4
        choose_test_kernel = 5
        test_kernels = [test_kernels[choose_test_kernel]]

    else:
        # test_kernels = [
            # '2mm',
            # '3mm',
            # 'adi',
            # 'atax',
            # 'atax-medium',
            # 'bicg',
            # 'bicg-large',
            # 'bicg-medium',
            # 'correlation',
            # 'covariance',
            # 'doitgen',
            # 'doitgen-red',
            # 'fdtd-2d',
            # 'fdtd-2d-large',
            # 'gemm-blocked',
            # 'gemm-ncubed',
            # 'gemm-p',
            # 'gemm-p-large',
            # 'gemver',
            # 'gemver-medium',
            # 'gesummv',
            # 'gesummv-medium',
            # 'heat-3d',
            # 'jacobi-1d',
            # 'jacobi-2d',
            # 'md',
            # 'mvt',
            # 'mvt-medium',
            # 'nw',
            # 'seidel-2d',
            # 'spmv-ellpack',
            # 'stencil',
            # 'stencil-3d',
            # 'symm',
            # 'symm-opt',
            # 'symm-opt-medium',
            # 'syr2k',
            # 'syrk',
            # 'trmm',
            # 'trmm-opt',
        # ]

        test_kernels = [
            # 'gemver',
            # 'fdtd-2d',
            # 'correlation',
            'adi',
        ]


        # 6 new larger kernels,
        test_kernels = [
        '3d-rendering',
        # 'att-3mm',
        # 'att-3mm-fuse',
        # 'knn',
        # 'spam-filter',
        # 'vmmv'
        ]

        if DEBUG:
            test_kernels = test_kernels[0:2]

    if not DEBUG:
        exp_name = 'HARP'
        if exp_name == '':
            raise RuntimeError(
                f'Please give an informative exp_name so that the person running HLS won\'t be confused'
            )
    else:
        exp_name = 'just_trying_randomly'


ignore_kernels = []

# ignore_kernels = ['3d-rendering', 'att-3mm', 'att-3mm-fuse', 'knn', 'spam-filter', 'vmmv']

# test_kernels = [test_kernels[5]]


# test_kernels1 = ['jacobi-1d', '3mm', 'fdtd-2d', 'gemm-p', 'gemver']
# test_kernels2 = ['fdtd-2d', 'jacobi-2d', 'trmm-opt'] ## to be used to split the kernels between training and testing. this is the list of test kernels
# test_kernels = list(set(test_kernels1 + test_kernels2))
# test_kernels = test_kernels2

# if 'vpn' in get_host():
#     # local machine
#     test_kernels = None  # will be fast
# # test_kernels = None # will be fast

if test_kernels is not None:
    tvt_split_by = 'kernels_inductive'
    val_ratio_in_test_kernels = 0  # what percentage to allocate for validation
    val_ratio_in_train_kernels = 0.15  # what percentage to allocate for validation
    # what percentage to allocate for additional transductive testing
    test_ratio_in_train_kernels = 0.15
    # shuffle_val_test = False
    # shuffle_val_test = True
    # parser.add_argument('--shuffle_val_test', type=bool, default=shuffle_val_test)

    if HARP_setup:  # 90:5:5; no holdout test (just transductive test)
        val_ratio_in_test_kernels = 0
        val_ratio_in_train_kernels = 0.05
        test_ratio_in_train_kernels = 0.05
        if DEBUG:
            val_ratio_in_train_kernels = 0.15
            test_ratio_in_train_kernels = 0.15  # otherwise no test...
else:
    tvt_split_by = 'designs_transductive'
    val_ratio = 0.15  # ratio of database for validation set


itype_mask_perc = 0
# itype_mask_perc = 0.15


gtype = 'programl'


# only_pragma_nodes = True # dangerous; will only keep pragma nodes and no edge_index
only_pragma_nodes = False


# encode_full_text = 'word2vec'
# encode_full_text = 'roberta'
encode_full_text = 'None'


fulltext_dim = None
if encode_full_text == 'word2vec':
    # fulltext_dim = 8
    fulltext_dim = 16
    # fulltext_dim = 32
    # fulltext_dim = 64
    # fulltext_dim = 128


pairwise_class = None

eval_pairwise = True


if task in ['regression', 'class']:

    # pairwise_class = False
    pairwise_class = True

    if pairwise_class:
        assert task == 'regression'

        # pairwise_batch_size = 32
        # pairwise_batch_size = 200
        # pairwise_batch_size = 256 * 5
        # pairwise_batch_size = 256
        pairwise_batch_size = 768

        pairwise_batch_size = 512

        # # reduced mem mode
        # pairwise_batch_size = 64

        if sequence_modeling:
            # pairwise_batch_size = 64
            pairwise_batch_size = 32
            if 'codellama' in code_encoder:
                pairwise_batch_size = 1
        if subtask == 'inference':
            pairwise_batch_size = 64  # so less memory usage
        # pairwise_batch_size = 512

        # if DEBUG:
        #     pairwise_batch_size = 1

        write_pairs_to_disk = False
        # write_pairs_to_disk = True # may take a lot of disk space

        pairwise_what_branches = ['regression']  # vanilla
        # pairwise_what_branches = ['regression', 'pariwise_comparison']
        # pairwise_what_branches = ['pariwise_comparison'] # deprecated

        if 'pariwise_comparison' in pairwise_what_branches:

            # comp_ops = ['hadamard', 'diff', 'emb_d1', 'emb_d2']
            comp_ops = ['hadamard', 'diff', 'emb_d1', 'emb_d2', 'node_att_diff']

            # loss_components = 'both'
            # loss_components = 'regression_only'
            loss_components = 'class_only'  # no regular data loader

            if 'node_att_diff' in comp_ops:
                loss_components = 'class_only'  # this operation only supports comparing designs of the same kernel
                # loss_components = 'both'  # if must do both, then will not do anything when requested to forward_pairwise

                # node_att_diff_cat_both = False # old
                node_att_diff_cat_both = True  # new

                # node_att_diff_scaled_att = True  # old
                node_att_diff_scaled_att = False  # new

                if 'inference' in subtask:
                    cuda_synchronize_trick = True  # may be needed to avoid weird error "RuntimeError: CUDA error: an illegal memory access was encountered" in node_att_diff.py


            if loss_components == 'both':
                fix_encoder_classMLPs = False
                # fix_encoder_classMLPs = True

        if subtask == 'train':
            if 'pariwise_comparison' in pairwise_what_branches:
                # pairwise_train_schemes = ['per_batch']
                pairwise_train_schemes = ['dedicated']
                # pairwise_train_schemes = ['per_batch', 'dedicated']

        elif subtask == 'inference':
            # eval_pairwise = False
            eval_pairwise = True

        branch_discrepancy_loss = False
        # branch_discrepancy_loss = True # only works if both branches are there in the architecture...

        listmle_loss = False
        # listmle_loss = True # sort of deprecated

        pairwise_loss = 'cross_entropy'
        # pairwise_loss = 'ranknet_v1' # P_true_{ij} = 1/0, P_pred_{ij} = sigmoid(o1 - o2)
        # pairwise_loss = 'ranknet_v2' # 1 - sigmoid((y1 - y2) * (o1 - o2))
        # pairwise_loss = 'ranknet_v1+mse'
        # pairwise_loss = 'ranknet_v2+mse'
        # pairwise_loss = 'cross_entropy+mse'
        # pairwise_loss = 'padr' # Performance-Aligned Differentiable Ranking (PADR) Loss

        # Be careful! Now we are no longer loading the FLAGS. So if
        # we set pairwise_loss or rank_score_deduction_cross_entropy to something wrong
        # i.e. mismatch with training-time loss FLAGS,
        # then in inference and dse we will have inconsistent results.
        if pairwise_loss == 'cross_entropy':
            # rank_score_deduction_cross_entropy = 'softmax'
            rank_score_deduction_cross_entropy = (
                'softmax_0.5_to_hard'  # softmax --> thresholding --> 0/1
            )

        # transitivity_loss = False
        # # transitivity_loss = True # TODO

        # self_comp_loss = False
        # # self_comp_loss = True # TODO

        # loss_components_to_keep = 'all'

        loss_components_to_keep = ['perf', 'perf_pairwise_class']

        # loss_components_to_keep = ['perf_pairwise_class']

        # pairwise_loss_mul_factor = 1 # default
        # pairwise_loss_mul_factor = 0.125
        # pairwise_loss_mul_factor = 0.25
        # pairwise_loss_mul_factor = 0.5
        # pairwise_loss_mul_factor = 2
        # pairwise_loss_mul_factor = 4
        pairwise_loss_mul_factor = 8
        

        if 'ranknet' in pairwise_loss:  # tricky: new experiments ignoring perf loss
            # Earlier, with the cross_entropy loss,
            # we sort of keep both branches and train both.
            # New experiments still create both branches but only train the pairwise branch.
            loss_components_to_keep = [
                'perf_pairwise_class'
            ]  # pure pairwise model without regression; tricky: in reality, we still create the MLPs for regression in the model arch but we never train that branch; and in test() we still report the RMSE but we should ignore it (it is big probably)
        elif 'padr' in pairwise_loss:
            loss_components_to_keep = [
                'perf',
                'perf_pairwise_class',
            ]  # should do both to guide the model (seeming to have more stable loss)

        if pairwise_what_branches == ['regression']:
            pairwise_loss = 'None'
            loss_components_to_keep = 'all'

        if (
            subtask == 'train'
            and 'pariwise_comparison' in pairwise_what_branches
            and 'dedicated' in pairwise_train_schemes
        ):

            # symmetry_aug = False
            symmetry_aug = True

            # format: (<diff_by>, <epoch id>, <max num pairs to save time>)
            # train_time_pragma_differing_by = [
            #     (1, 0, 500),
            #     (2, 400, 500),
            #     (3, 800, 500),
            #     ('all', 1200, 500),
            # ]

            # train_time_pragma_differing_by = [
            # (1, 0, 1000),
            # ('all', 1200, 1000),
            # ]

            train_time_pragma_differing_by = [
                ('all', 0, 1000),
            ]

            # train_time_pragma_differing_by = [
            #     ('all', 0, 500),  # quicker
            # ]

            # train_time_pragma_differing_by = 2
            # train_time_pragma_differing_by = 3
            # train_time_pragma_differing_by = 'all'
            if DEBUG:
                train_time_pragma_differing_by = [(1, 0, 5), ('all', 3, 7)]
                # train_time_pragma_differing_by = [(1, 0, 7)]
                # train_time_pragma_differing_by = [('all', 0, 5)]

        elif subtask == 'inference':
            skip_pointwise_infer = False  # both
            # skip_pointwise_infer = True  # only pairwise; quicker

            # inference_time_pragma_differing_by_li = [1]

            # inference_time_pragma_differing_by_li = ['>=2']

            # inference_time_pragma_differing_by_li = [1, 2, 3, '>=4']
            # inference_time_pragma_differing_by_li = [2]
            # inference_time_pragma_differing_by_li = [3]
            inference_time_pragma_differing_by_li = [1, 2, 3, 'all']
            # inference_time_pragma_differing_by_li = []  # to be quick

            if DEBUG:
                # inference_time_pragma_differing_by_li = [1, 2, 3]
                inference_time_pragma_differing_by_li = ['all']

        if type(loss_components_to_keep) is list:
            assert len(loss_components_to_keep) >= 1

    else:
        eval_pairwise = False


load_model_class = None
prune_invalid = None
if 'dse' in subtask:
    if task != 'regression':
        raise RuntimeError(
            f'Must set task to regression when doing DSE! Can specify load_model_class below though for pruning invalid designs using an auxiliary classification model'
        )

    explorer = 'exhaustive'
    dist_parent = True
    dist_child = True

    prune_util = True  # only DSP and BRAM
    prune_class = False

    ordered_pids = [
        '__PARA__L3',
        '__PIPE__L2',
        '__PARA__L1',
        '__PIPE__L0',
        '__TILE__L2',
        '__TILE__L0',
        '__PARA__L2',
        '__PIPE__L0',
    ]

    separate_perf = False

    # dse_mode = 'gen_embeddings'
    dse_mode = 'real_dse'

    collect_embs = True

    load_model_class = '<classification_model_path>'
    if load_model_class != 'None':
        load_model_class = check_prepend_root_folder(load_model_class)
        if '.pth' not in load_model_class and 'pt' not in load_model_class:
            load_model_class += '/val_model_state_dict.pth'
        if task != 'regression':
            raise RuntimeError(
                f'Must set task to regression when load_model_class is not None'
            )

        # Codde below only works with/is specially written for Atefeh's HARP classifier.
        if DEBUG:
            debug_str = '_DEBUG'
        else:
            debug_str = ''
        # class_model_save_dir = f'/home/yba/software-gnn/save/programl/v21_True-1-epbch-class_False_False_speedup-log2_False_whole-machsuite-poly{debug_str}_programl_False_False_None_None_nosplit_None_True'

        # prune_invalid = False
        prune_invalid = True

        # if DEBUG:
        #     prune_invalid = False  # so that always have something to run for stage 2

        load_model_weights = (
            True  # may override previously defined `load_model_weights`
        )

    else:
        prune_invalid = False
        assert (
            not prune_invalid
        ), 'Should NOT prune designs with a randomly initialized model -- Just set prune_invalid tp False'

    dse_timeout = 12 * 60 * 60  # 12 hours

    # dse_timeout *= 3 # TODO!

    if DEBUG:
        dse_timeout = 20  # quick

    # max_num_explored_points = 75000
    max_num_explored_points = float('inf')

    # max_num_explored_points *= 3
    if DEBUG:
        # max_num_explored_points = 500
        max_num_explored_points = float('inf')

    num_top_designs = 10

    # pairwise_comp_DSE = False
    pairwise_comp_DSE = True

    if pairwise_comp_DSE:
        assert pairwise_class

        # num_designs_to_cache_during_search = float('inf') # risky; may explore the disk space size...
        num_designs_to_cache_during_search = (
            5000  # kind of like saving a lot of objects to disk,,,
        )

        num_cand_for_ver = 100

        load_model_pairwise = f'<log_folder>'
        load_model_pairwise = check_prepend_root_folder(load_model_pairwise)
        if '.pth' not in load_model_pairwise and 'pt' not in load_model_pairwise:
            load_model_pairwise += '/val_model_state_dict.pth'

        # cuda_synchronize_trick = False
        cuda_synchronize_trick = True  # may be needed to avoid weird error "RuntimeError: CUDA error: an illegal memory access was encountered" in node_att_diff.py

        assert 'perf_pairwise_class' in loss_components_to_keep

        # how_to_take_final = 'score_sorting'
        how_to_take_final = 'ranked_choice_voting'

if not prune_invalid:
    load_model_class = 'None'

# MAML = True
MAML = False

if subtask == 'visualize':

    mode = 'vis_dse'
    # mode = 'vis_trained_model'

    if mode == 'vis_dse':
        # Sep P T.
        # dse_collection_path = 'src/logs/regression_dse_whole-machsuite-polly_2022-04-29T14-25-01.158127_3mm_MAML-13'
        # dse_kernel = '3mm'
        # dse_collection_path = 'src/logs/regression_dse_whole-machsuite-polly_2022-04-29T14-26-06.706674_fdtd_MAML-13'
        # dse_kernel = 'fdtd-2d'
        # dse_collection_path = 'src/logs/regression_dse_whole-machsuite-polly_2022-04-29T14-24-31.022782_gemmp_MAML-13'
        # dse_kernel = 'gemm-p'
        # dse_collection_path = 'src/logs/regression_dse_whole-machsuite-polly_2022-04-29T14-24-41.584875_gemver_MAML-13'
        # dse_kernel = 'gemver'
        dse_collection_path = 'src/logs/regression_dse_whole-machsuite-polly_2022-04-29T14-21-59.130876_jacob_MAML-13'
        dse_kernel = 'jacobi-1d'

        dse_collection_path = check_prepend_root_folder(dse_collection_path)

        # clustering = 'None'
        # clustering = 'K_Medoid'
        clustering = 'K_Means'

        if clustering in ['K_Medoid', 'K_Means']:
            n_clusters = 20

            if clustering == 'K_Medoid':
                metric = 'cosine'

        vis_y = 'matched'

    elif mode == 'vis_trained_model':

        MAML = True
        # MAML = False

        if MAML:
            MAML_split_kernel = False

        # vis_what = 'att'
        vis_what = 'tsne'

        if vis_what == 'tsne':
            only_kernel = 'None'
            # only_kernel = '3mm'
            # only_kernel = 'aes'
            # only_kernel = 'fdtd-2d'

            # only_kernel = 'mvt'
            # only_kernel = 'gemm-blocked'
            # only_kernel = 'bicg'
            # only_kernel = 'stencil'

            # vis_emb = 'X'
            vis_emb = 'gemb'

            if vis_emb == 'gemb':
                vis_emb_P_or_T = 'P'
                # vis_emb_P_or_T = 'T'

            vis_y = 'kernel_name'
            # vis_y = 'target'

    else:
        assert False

    # vis_design_sampler = True
    vis_design_sampler = False

    if vis_design_sampler:
        # adapt_designs_sample_algo = 'random'
        # adapt_designs_sample_algo = 'KNN_X'
        # adapt_designs_sample_algo = 'KNN_gemb'
        # adapt_designs_sample_algo = 'greedycosinesmallest_gemb'
        adapt_designs_sample_algo = 'spreadcosine_gemb'

        # adapt_designs_sample_algo_quality = 'avg_dist'
        # adapt_designs_sample_algo_quality = 'avg_dot'
        adapt_designs_sample_algo_quality = 'avg_cosine'
        # adapt_designs_sample_algo_quality = 'volume'

        save_sampled_designs = True
        # save_sampled_designs = False


load_encoders_label = None
encoder_path = None
# encoder_path = '../save/harp/r13-ifdb-epbch-regression_ep-False_nowi_False-n_speedup-log2_np_False_whole-machsuite-poly_programl_False_False_None_None_nosplit_regular_encoder_True_s_penhance_codet5_64_tm_pk_v2_fc_co16_programl+src_code_feed_pclcc_pseudo_ntic_igt/encoders.klepto'


if load_pretrained_GNN:
    # encoder_path = '../../file/zongyue_pretrain/v18_v20_encoders.klepto'
    encoder_path = '../../file/zongyue_pretrain/harp_encoders.klepto'
    load_encoders_label = 'zongyue_04272023'


# load_model_HARPnet = False
load_model_HARPnet = True  # if turned on, will always use the HARPNet model class!
if sequence_modeling and not multi_modality:
    load_model_HARPnet = False

if (
    load_model_HARPnet
    and subtask == 'train'
    and eval_pairwise
    and 'pariwise_comparison' in pairwise_what_branches
):
    HARP_different_lrs = False
    # HARP_different_lrs = True

    base_lr = 5e-6
    pairwise_class_lr = 5e-4

encoder_path_class = 'None'
if (load_model is not None and 'atefeh' in load_model) or (
    load_model_class is not None
    and 'atefeh' in load_model_class
    or load_model_HARPnet  # comment this out later! continual training from HARP
):
    encoder_path = '<path_to>/regression-pragma_as_MLP-encoders.klepto-0.klepto'
    encoder_path_class = '<path_to>/models/v21_models/class-pragma_as_MLP-encoders.klepto-0.klepto'
    # encoder_path_class = '/home/yba/software-gnn/file/pragma_as_MLP-encoders.klepto'
    load_encoders_label = 'atefeh_harp'

if encoder_path != 'None' and (not (encoder_path is None)):
    encoder_path = check_prepend_root_folder(encoder_path)
if encoder_path_class != 'None' and (not (encoder_path_class is None)):
    encoder_path_class = check_prepend_root_folder(encoder_path_class)


# encoder_path = '../save/harp/r13-ifdb-epbch-regression_ep-False_nowi_False-n_speedup-log2_np_False_whole-machsuite-poly_programl_False_False_None_None_nosplit_regular_encoder_True_s_penhance_codet5_64_tm_pk_v2_fc_co16_programl+src_code_feed_pclcc_pseudo_ntic_igt/preprocessors.klepto'
# encoder_path = None


# outlier_removal = None
# outlier_removal = '0'
# parser.add_argument('--outlier_removal', default=outlier_removal)


model_tag = 'test'

activation = 'elu'


outlier_removal = None
# outlier_removal = '50'

no_pragma = False


if sequence_modeling:
    if combine_node_edge_labels:
        num_layers = 8
    else:
        # num_layers = 0 # default; fast
        # num_layers = 4
        num_layers = 8
        if interleave_GNN_transformer:  # TODO: tune it
            # num_layers = 0
            # num_layers = 6
            num_layers = 8

    if not multi_modality:  # seq only
        num_layers = 1  # just need the conv_first layer

        num_layers = 8  # to load older models, use this
else:
    if graph_transformer_option:
        # num_layers = 2
        num_layers = 8
    else:
        num_layers = 6

        # num_layers = 8

# if DEBUG:
#     num_layers = 3

D = 64  # codet5, GNN
# D = 128
# D = 256
# D = 512
# D = 768
# D = 1024

if multi_modality and combine_fashion == 'share_final_MLPs' and feed_p_to_tf:
    D = 512
if code_encoder in ['codebert', 'graphcodebert']:
    D = 768
elif 'codellama' in code_encoder:
    D = 1024
    # D = 128
elif code_encoder == 'TheBloke/CodeLlama-7B-GPTQ':
    D = 512
elif code_encoder is not None and 'codegen' in code_encoder:
    D = 1024
elif code_encoder == 'OpenAI':
    D = 768  # TODO: maybe this is a crticial hyperparameter that should be tuned!!!!!!!!!
    # num_hidden_lyr =


# if TASK == 'regression':
# target = 'quality'
# target = 'util-BRAM'
# target = 'util-DSP'
# target = 'util-LUT'
# target = 'util-FF'
# multi_target = ['perf', 'util-DSP']
target = ['perf']
# target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
# DAC'22
# multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
# multi_target = ['util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
# multi_target = ['util-DSP', 'util-BRAM', 'util-FF']
# multi_target = ['util-BRAM']
# multi_target = ['perf', 'util-DSP', 'util-BRAM']
# target = 'perf'
assert target[0] == 'perf'  # assumed by our code

if graph_transformer_option is None:
    # gnn_type = 'gcn'
    # gnn_type = 'gat'
    gnn_type = 'transformer'

# if latency is less than this, prune the point (used when synthesis is not valid)
min_allowed_latency = 100.0

encode_edge = True
if sequence_modeling and not multi_modality:
    encode_edge = False


encode_edge_position = False


# jkn_mode = 'lstm'
jkn_mode = 'max'


jkn_enable = True
# jkn_enable = False


node_attention = True
# if sequence_modeling:
#     node_attention = False


pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'  # keep both as 2l

pragma_as_MLP = True
# pragma_as_MLP = False

if not multi_modality and sequence_modeling:
    pragma_as_MLP = False


gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None
if pragma_as_MLP:
    # if gnn_layer_after_MLP == 1:
    #     model_ver = 'best_post-gnn-2l'

    if type_parallel == '2l':
        pragma_MLP_hidden_channels = '[in_D // 2]'
    elif type_parallel == '3l':
        pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'

    if type_merge == '2l':
        merge_MLP_hidden_channels = '[in_D // 2]'
    elif type_merge == '3l':
        merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
    else:
        raise NotImplementedError()
    (
        gae_T,
        P_use_all_nodes,
        separate_pseudo,
        separate_T,
        dropout,
        num_features,
        edge_dim,
    ) = (False, True, True, False, 0.1, 153, 335)
    separate_P = True

else:
    (
        gae_T,
        P_use_all_nodes,
        separate_pseudo,
        separate_T,
        dropout,
        num_features,
        edge_dim,
    ) = (True, False, False, True, 0.1, 156, 335)
    separate_P = True

    # model_ver = 'hierarchy-PT'
if pragma_as_MLP:
    assert graph_type == 'extended-pseudo-block-connected-hierarchy'
    # assert multi_modality, 'Not implemented yet for basic non-multi_modality (need to do that in model.py\'s Net'
    # number of message passing layers after MLP (pragma as MLP)

    pragma_as_MLP_list = ['tile', 'pipeline', 'parallel']

    pragma_scope = 'block'

    keep_pragma_attribute = False if pragma_as_MLP else True

    pragma_order = 'sequential'
    pragma_order = 'parallel_and_merge'  # best result

    # pragma_MLP_hidden_channels = None
    # pragma_MLP_hidden_channels = '[in_D // 2]'

    # merge_MLP_hidden_channels = '[in_D // 2]'

    num_conv_layers_for_MLP_pragma = 1


MLP_common_lyr = 3
MLP_individual_lyr = 7

if 'atefeh' in load_model or load_model_HARPnet:
    MLP_common_lyr = 0

if node_attention:
    node_attention_MLP = False

    # separate_P_T = True
    # critical: for NeurIPS 2023, consistently turn off P/T separation for all models
    separate_P_T = False
    if sequence_modeling:
        separate_P_T = False
    # separate_P = True

    separate_icmp = False


pragma_encoder = True
if dataset == 'simple-programl' or target_kernel is not None:
    pragma_encoder = False
pragma_uniform_encoder = True


# separate_pseudo = True
# parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

epsilon = 1e-3

normalizer = 1e7
util_normalizer = 1

# max_number = 3464510.00
max_number = 1e10


# 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
norm_method = 'speedup-log2'


target_preproc = 'None'
# target_preproc = 'minmax'
# target_preproc = 'z-score'

target_convert_back = True

invalid = False  # False: do not include invalid designs

multi_target = True

activation_type = 'gelu'


# For ranking.
# margin_loss = True
margin_loss = False


save_model = True


if save_model:
    save_every_epoch = 10000  # do not save too many models!

encode_log = False


target_factor = 1
# target_factor = 100
# target_factor = 1e-7
# target_factor = 1e-5


target_transform = None
# target_transform = 'log'


loss_scale = {
    'perf': 1.0,
    'util-DSP': 1.0,
    'util-BRAM': 1.0,
    'util-LUT': 1.0,
    'util-FF': 1.0,
}
# loss_scale = None

if task == 'rl':
    num_envs = 2


# batch_size = 2
# batch_size = 128
# batch_size = 64
# default; 0 means that the data will be loaded in the main process
data_loader_num_workers = 0
if sequence_modeling:
    if data_repr == 'ast':
        batch_size = 1
        data_loader_num_workers = 0
    else:
        if not multi_modality:
            # batch_size = 4
            # batch_size = 8

            # batch_size = 16

            # batch_size = 64

            batch_size = 32

            # batch_size = 1024

            # batch_size = 128

            if pairwise_class:
                batch_size = 16

                batch_size = 64

            if 'codebert' in code_encoder:
                batch_size = 2
            elif 'codellama' in code_encoder:
                batch_size = 1
            elif code_encoder == 'OpenAI':
                batch_size = 5012 * 2

            if task == 'class':
                batch_size = 2
                # batch_size = 4

        else:
            if 'codebert' in code_encoder:
                batch_size = 1
            else:
                batch_size = 2
                batch_size = 4
                # batch_size = 8
                if chunk_offset > 0:
                    # batch_size = 5
                    if get_host() == 'Scai1':
                        batch_size = 2
                    if get_host() == 'Scai2':
                        batch_size = 2
    if 'codegen' in code_encoder:
        batch_size = 1

    if full_quadratic_attention:
        batch_size = 1

else:
    if graph_transformer_option is not None:
        batch_size = 2
    else:

        # batch_size = 1024
        # batch_size = 512 * 12
        batch_size = 512 * 3
        # batch_size = 500
        # batch_size = 256
        # batch_size = 128
        # batch_size = 64
        if task == 'class':
            batch_size = 8
        # batch_size = 8

        # # reduced mem mode
        batch_size = 128

        if subtask == 'inference':
            batch_size = 128  # save memory

pin_memory = True
if 'dse' in subtask:
    pin_memory = False

# if use_peft:
#     batch_size *= 2
if 'vpn' in get_host():
    # local machine
    # batch_size = 1
    data_loader_num_workers = 0


if get_host() == 'yba':
    gpu = 0

    # gpu = 1

    if gpu == 1:
        gpu = 0
    elif gpu == 0:
        gpu = 1
else:
    # gpu = 0
    # gpu = 1
    # gpu = 2
    # gpu = 3
    # gpu = 4
    # gpu = 5
    # gpu = 6
    # gpu = 7  # scai5 does not have it
    # gpu = 'auto'
    gpu = 'user_input'  # need to manually enter in terminal
    if DEBUG:
        gpu = 'auto'  # to be quick; wi
        # gpu = 3  # even quicker than 'auto'
        # gpu = 'user_input'

    if get_host() == 'ipp1-trident-01':
        no_touch_gpus = [0, 1]  # for other lab mates
    else:
        no_touch_gpus = []
    if torch.cuda.is_available() and gpu in ['auto', 'user_input']:
        gpu = get_best_gpu(gpu, no_touch_gpus)

device = str(
    'cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu'
)


if task == 'regression':
    # epoch_num = 1
    # epoch_num = 800
    # epoch_num = 1000

    # epoch_num = 1500
    epoch_num = 1600
else:
    epoch_num = 200
if 'vpn' in get_host():
    epoch_num = 0  # local laptop
    if force_regen:
        epoch_num = 0

if DEBUG:
    epoch_num = 10

    epoch_num = 100

# Will early stop!
max_stagnant_epochs = 10000


debug_iter = -1  # no debugging
# debug_iter = 2
# debug_iter = 9999


# if subtask == 'train':

if 'train' in subtask:

    # ignore_testing = False # default; slow
    ignore_testing = True

    # if pairwise_class:
    #     ignore_testing = False # just ignore testing to be quick

    ignore_validation = False  # default; slow
    if 'vpn' in get_host():
        ignore_validation = True


if 'inference' in subtask:
    assert replace_with_random_weights in [None, False]

    save_emb = False  # default
    # save_emb = True

    adaptation_needed = False  # quicker; do it when adaptation takes time, e.g. for CodeLlama/OpenAI's GPT
    # adaptation_needed = True

    if model == 'OpenAI':
        adaptation_needed = False  # no adaptation needed

    if adaptation_needed:

        adaptation_valid_num = 40

        repeat_times = 1
        if 'vpn' in get_host():  # local
            repeat_times = 1
        # if 'dse' in subtask:
        #     repeat_times = 2

        # adaptation_num_dp = 20
        adaptation_num_dp = 100  # sample more!
        # adaptation_num_dp = 200
        # adaptation_num_dp = 'all'
        # adaptation_num_dp = 1
        if DEBUG:
            adaptation_num_dp = 1

        # num_mini_epochs = 10
        # num_mini_epochs = 100  # run more epochs!

        num_mini_epochs = 150
        # num_mini_epochs = 1

        # adaptation_num_dp = 300

        # num_mini_epochs = 10
        # num_mini_epochs = 30

        if 'vpn' in get_host():  # local
            num_mini_epochs = 1
        if DEBUG:
            num_mini_epochs = 1

        if num_mini_epochs <= 10:
            test_which_adapted_model = 'last_epoch'
        else:
            test_which_adapted_model = 'best_train'

        if adaptation_num_dp > 50:
            test_which_adapted_model = 'best_valid'


shuffle = True  # default: shuffle train data
# shuffle = False


weight_switch = False
# parser.add_argument('--weight_switch', type=bool, default=weight_switch)
# opt_type = 'Adam'
opt_type = 'AdamW'


# lr = 0.001
# lr = 1e-4
# lr = 1e-5
if sequence_modeling:
    lr = 1e-5
    # lr = 5e-5
    # lr = 5e-4
    # lr = 1e-4
    # lr = 1e-6
    if 'codegen' in code_encoder:
        # lr = 1e-3
        # lr = 1e-4
        # lr = 1e-6
        # lr = 5e-6
        lr = 5e-7
    elif 'codellama' in code_encoder:
        # lr = 1e-5
        lr = 1e-5
    elif code_encoder == 'OpenAI':
        lr = 5e-4
else:
    if graph_transformer_option is None:
        # lr = 5e-4
        # lr = 5e-3
        lr = 1e-3
        # lr = 5e-4
        # lr = 5e-6
        # lr = 1e-6
    else:
        lr = 1e-4

if (
    load_model_HARPnet
    and subtask == 'train'
    and eval_pairwise
    and 'pariwise_comparison' in pairwise_what_branches
    and HARP_different_lrs
):
    lr = None  # overriden by base_lr etc.

max_grad_norm = None
# max_grad_norm = 1.0


# lr_scheduler_type = 'linear'
lr_scheduler_type = 'cosine'
# https://github.com/huggingface/transformers/issues/20552 -- perhaps just use cosine or linear...
# maybe we need to specify num_cycles in addition https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
# lr_scheduler_type = 'cosine_with_restarts'
# lr_scheduler_type = 'polynomial'
# lr_scheduler_type = 'constant'
# lr_scheduler_type = 'constant_with_warmup'

num_warmup_steps = 0
max_train_steps = None
# See https://huggingface.co/docs/transformers/v4.18.0/en/performance 'Gradient Accumulation' section
# gradient_accumulation_steps = 128
gradient_accumulation_steps = 1

if 'AdamW' in opt_type:
    # weight_decay = 0.01  # default of Adam

    # weight_decay = 0.0001  # even smaller
    weight_decay = 0
else:
    weight_decay = 0
# elif opt_type == 'AdamW':
#    weight_decay = 0.01  # default of AdamW
# else:
#    assert False
# weight_decay = 0.1 # be careful; large L2 regularization


# plot_pred_points = True
plot_pred_points = False
if SSL:
    plot_pred_points = False


fix_randomness = True  # TODO: may have issue with result
# fix_randomness = False
if fix_randomness:
    random_seed = 123
    # random_seed = 777
user = get_user()
hostname = get_host()


###################################### Below: no need to touch ######################################

try:
    import git
except Exception as e:
    raise type(e)(
        f'{e}\nYunsheng: Run pip install gitpython or\nconda install gitpython'
    )
repo = git.Repo(Path(__file__).parent.parent.absolute())
repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
try:
    local_branch_name = repo.active_branch.name
except:
    local_branch_name = 'DETACHED_' + repo.head
commit_sha = repo.head.object.hexsha

from os.path import dirname, abspath

proj_dir = dirname(dirname(abspath(__file__)))

vars = OrderedDict(vars())
FLAGS = OrderedDict()
for k, v in vars.items():
    if not k.startswith('__') and type(v) in [
        int,
        float,
        str,
        list,
        dict,
        type(None),
        bool,
    ]:
        FLAGS[k] = v
FLAGS = SimpleNamespace(**FLAGS)
