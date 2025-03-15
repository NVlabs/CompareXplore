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

from graph_transfotmer import get_num_features_graph_transformer, GPSConv
from data_src_code import create_code_encoder
from pairwise import get_comp_result_tagret
from data import get_num_features
from nn import MyGlobalAttention
from node_att_diff import NodeAttDiff, split_vec_mat_into_2_halves
from config import FLAGS
from saver import saver
from utils_nn import create_graph_att_module
from utils import MLP, create_act, estimate_model_size, MLP_multi_objective, deduce_MLP_dimensions
from pairwise import PairwiseLoss

import data

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    JumpingKnowledge,
    TransformerConv,
    GCNConv,
    GINEConv,
)
from torch_geometric.nn import global_add_pool
import torch.nn as nn

from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU

from collections import OrderedDict
import math

class Model(nn.Module):
    def _create_src_code_bert(self):
        # saver.log_info('Creating bert_model')
        bert_model, BERT_DIM = create_code_encoder()
        # saver.log_info('Created bert_model')
        # estimate_model_size(
        #     bert_model, f'bert_model {FLAGS.code_encoder}', saver)
        return bert_model, BERT_DIM

    def _get_GNN_conv_class(self):
        if FLAGS.graph_transformer_option is not None:
            conv_class = None
        else:
            if FLAGS.gnn_type == 'gat':
                conv_class = GATConv
            elif FLAGS.gnn_type == 'gcn':
                conv_class = GCNConv
            elif FLAGS.gnn_type == 'transformer':
                conv_class = TransformerConv
            else:
                raise NotImplementedError()
        return conv_class

    def _create_conv_first(
        self,
        edge_dim,
        in_channels,
        D,
        conv_class,
        sequence_modeling,
        hidden_channels=None,
    ):
        if sequence_modeling:
            if hidden_channels is None:
                hidden_channels = [self.bert_dim // 2, self.bert_dim // 4]
            conv_first = MLP(
                in_channels,
                D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,  # to load older models, turn this on
                num_hidden_lyr=2,
            )
        else:
            if FLAGS.graph_transformer_option is not None:
                gt_dim = get_num_features_graph_transformer(in_channels)
                conv_first = MLP(
                    gt_dim, D, activation_type=FLAGS.activation, num_hidden_lyr=2
                )
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    # print(in_channels)
                    conv_first = conv_class(in_channels, D, edge_dim=edge_dim)
                else:
                    conv_first = conv_class(in_channels, D)
        return conv_first

    def _create_GNN_conv_layers(self, num_layers, edge_dim, D, conv_class):
        conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            if FLAGS.graph_transformer_option is not None:
                gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
                if gt_conv_type == 'mha':
                    conv = nn.MultiheadAttention(
                        D,
                        num_heads=FLAGS.graph_transformer_option['num_heads'],
                        batch_first=True,
                    )
                elif gt_conv_type == 'gps_conv':
                    if FLAGS.graph_transformer_option['need_local_mpnn']:
                        local_mpnn = GINEConv(
                            Sequential(
                                Linear(D, D), create_act(FLAGS.activation), Linear(D, D)
                            ),
                            edge_dim=edge_dim,
                        )
                    else:
                        local_mpnn = None
                    conv = GPSConv(
                        D,
                        conv=local_mpnn,
                        heads=FLAGS.graph_transformer_option['num_heads'],
                    )
                else:
                    raise NotImplementedError()
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    conv = conv_class(D, D, edge_dim=edge_dim)
                else:
                    conv = conv_class(D, D)
            conv_layers.append(conv)
        jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)
        return conv_layers, jkn

    def _get_target_list(self):
        if 'regression' in self.task:
            _target_list = FLAGS.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.task]
            # if FLAGS.new_speedup == False:
            #     self.target_list = [t for t in _target_list if t != 'perf' else 'kernel_speedup'] # to use with trained model from old speedup
            # else
            target_list = [t for t in _target_list]
        else:
            target_list = ['perf']
        return target_list

    def _create_decoder_MLPs(
        self, input_dim, D, target_list, out_dim, hidden_channels=None
    ):
        MLPs = None
        MLPs_pairwise_class = None
        # if FLAGS.node_attention:
        #     if FLAGS.separate_P_T:
        #         in_D = 2 * input_dim
        #     else:
        #         in_D = input_dim
        # else:
        in_D = input_dim
        if hidden_channels is None:

            if D > 64:
                hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
            else:
                hidden_channels = [D // 2, D // 4, D // 8]

            # hidden_channels = []
            # D_hidden = input_dim
            # for _ in range(FLAGS.MLP_individual_lyr):
            #     D_hidden = max(D_hidden // 2, out_dim)  # each time reduce by half
            #     hidden_channels.append(D_hidden)

        if not FLAGS.SSL:
            self.MLP_out_dim = out_dim
            MLPs = self._create_target_MLPs_dict(
                target_list, hidden_channels, in_D, out_dim
            )



            if (
                FLAGS.pairwise_class
                and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            ):
                in_D_class = in_D * len(FLAGS.comp_ops)
                MLPs_pairwise_class = self._create_target_MLPs_dict(
                    self.target_list, hidden_channels, in_D_class, 2
                )  # binary


            # if (
            #     FLAGS.pairwise_class
            #     and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            # ):
            #     raise RuntimeError(f'Deprecated! Check HARPNet. Need to consider loss_components_to_keep etc.')
            #     in_D_class = in_D * len(FLAGS.comp_ops)
            #     # TODO: choice: compare before each target or for/within/dedicated to each target
            #     MLPs_pairwise_class = self._create_target_MLPs_dict(
            #         target_list, hidden_channels, in_D_class, 2
            #     )  # binary
        return MLPs, MLPs_pairwise_class

    def _create_target_MLPs_dict(self, target_list, hidden_channels, in_D, out_D):

        if (not hasattr(self, "MLP_version")) or self.MLP_version == 'single_obj':
            MLPs = nn.ModuleDict()
            for target in self.target_list:
                MLPs[target] = MLP(
                    in_D,
                    out_D,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    num_hidden_lyr=len(hidden_channels),
                )
        else:
            MLPs = MLP_multi_objective(
                in_D,
                out_D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                objectives=self.target_list,
                num_common_lyr=FLAGS.MLP_common_lyr,
            )

        """ #commeted by zongyue
        rtn = nn.ModuleDict()
        
        for target in target_list:
            rtn[target] = MLP(in_D, out_D, activation_type=FLAGS.activation,
                              hidden_channels=hidden_channels,
                              num_hidden_lyr=len(hidden_channels))
        return rtn
        """
        return MLPs

    def _create_loss(self):
        if self.task == 'regression':
            out_dim = 1
            loss_function = nn.MSELoss(reduction='none')
        else:
            out_dim = 2
            loss_function = nn.CrossEntropyLoss()
        return out_dim, loss_function

    def _apply_bert(self, x, data, additional_x=None, nti_objs=None):
        x_shape = x.shape
        attention_mask = None
        if FLAGS.token_att_masking:
            attention_mask = data.attention_mask
        if hasattr(FLAGS, 'vis_transformer_att') and FLAGS.vis_transformer_att:
            output_attentions = True
        else:
            output_attentions = False
        if FLAGS.code_encoder == 'codet5':
            x_bert = self.bert_model(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                additional_x=additional_x,
                nti_objs=nti_objs,
            )
        elif FLAGS.code_encoder == 'OpenAI':
            x_bert = self.bert_model(data)
        else:
            x_bert = self.bert_model(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )
        if hasattr(FLAGS, 'vis_transformer_att') and FLAGS.vis_transformer_att:
            from data_src_code import tokenizer
            from bertviz import head_view, model_view

            attentions = x_bert['attentions']
            # Just vis the 0th sequence.
            attention = tuple(
                torch.unsqueeze(a[0], dim=0) for a in attentions
            )  # taking the 0th seq/chunk's attention maps
            tokens = tokenizer.convert_ids_to_tokens(x[0])
            saver.log_info(f'tokens={tokens}')
            for layer_id, a in enumerate(attention):
                assert a.shape[0] == 1
                num_heads = a.shape[1]
                for head_id in range(num_heads):
                    vec = a[0][head_id][0]
                    top_10_ids = torch.argsort(vec, descending=True)[0:10]
                    top_10_tokens = [tokens[token_id] for token_id in top_10_ids]
                    saver.log_info(
                        f'Token {tokens[0]}\'s top 10 attention tokens layer {layer_id} head {head_id}: {top_10_tokens}'
                    )
            f = ' '.join(tokens)
            saver.log_info(f'full string={f}')
            html_head_view = head_view(attention, tokens, html_action='return')
            with open(f"{saver.get_obj_dir()}/head_view.html", 'w') as file:
                file.write(html_head_view.data)
            saver.log_info(f'vis_transformer_att done; exiting')
            exit()

        if FLAGS.code_encoder == 'OpenAI':
            # OpenAI akready returns one embedding per design. No need to further extract/pool.
            x_bert_rtn = x_bert
        else:
            if FLAGS.chunk_emb == 'pooler':
                x_bert_rtn = x_bert['pooler_output']
            elif FLAGS.chunk_emb == 'cls':
                #             from data_src_code import tokenizer
                #             saver.log_info(f'x_bert={x_bert}')
                #             xxx = tokenizer("Hello, my dog is cute", return_tensors="pt")
                # q            yyy = self.bert_model(**xxx)
                #             saver.log_info(yyy)

                #             saver.log_info(x_bert)
                #             saver.log_info(f'last_hidden_state')
                #             saver.log_info(x_bert.last_hidden_state)
                #             saver.log_info(x_bert.last_hidden_state.shape)
                #             exit()

                x_hidden = x_bert.get('last_hidden_state')
                if x_hidden is None:
                    x_hidden = x_bert['hidden_states']
                    # saver.info(f'len(x_hidden)={len(x_hidden)}')
                    x_hidden = x_hidden[-1]  # last layer's hidden states
                    # saver.info(f'x_hidden.shape={x_hidden.shape}')
                x_bert_rtn = x_hidden[:, 0, :]  # 0th token
            else:
                assert False
        assert x_bert_rtn.shape == (
            x_shape[0],
            self.bert_dim,
        ), f'x_bert_rtn.shape={x_bert_rtn.shape}; (x_shape[0], self.bert_dim)={(x_shape[0], self.bert_dim)}; data.batch={data.batch}'
        # return x_bert_rtn.float(), x_bert

        if 'codellama' in FLAGS.code_encoder:
            saver.log_info_once(f'x_bert_rtn.dtype={x_bert_rtn.dtype}')
            x_bert_rtn = x_bert_rtn.float()
            saver.log_info_once(f'x_bert_rtn.float() due to codellama producing BFloat16; x_bert_rtn.dtype={x_bert_rtn.dtype}')
        return x_bert_rtn, x_bert

    def _get_act_func(self):
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()
        return activation

    def _apply_conv_first(self, conv_first, x, edge_index, edge_attr, activation):
        if FLAGS.graph_transformer_option is not None:
            out = conv_first(x)
        else:
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                out = conv_first(x, edge_index, edge_attr=edge_attr)
            else:
                # try:
                if len(edge_index) == 0:
                    raise ValueError(f'len(edge_index) == {len(edge_index)}; skip GNN')
                else:
                    # if FLAGS.sequence_modeling:
                    #     out = x
                    # else:
                    out = conv_first(x, edge_index)
                # except Exception as e:
                #     print(e)
        if activation is not None:
            out = activation(
                out
            )  # this is tricky! MLP does not apply activation for the last layer, but here we may apply it at the end
        return out

    def _apply_conv_layers(
        self,
        out,
        conv_layers,
        jkn,
        outs,
        edge_index,
        edge_attr,
        batch,
        data,
        activation,
    ):
        for i, conv in enumerate(conv_layers):
            if FLAGS.graph_transformer_option is not None:
                gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
                if gt_conv_type == 'mha':
                    attention_map_aug = FLAGS.graph_transformer_option.get(
                        'attention_map_aug'
                    )
                    if attention_map_aug is not None:
                        attn_mask = data.att_mask_tensor
                    else:
                        attn_mask = None
                    out, _ = conv(query=out, key=out, value=out, attn_mask=attn_mask)
                elif gt_conv_type == 'gps_conv':
                    out = conv(out, edge_index, batch, edge_attr=edge_attr)
                else:
                    raise NotImplementedError()
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    if edge_attr is not None:
                        # to ensure binary vectors
                        assert (
                            edge_attr.max() == 1
                        ), f'edge_attr wrong! Should be binary; max() is {edge_attr.max()}'
                    out = conv(out, edge_index, edge_attr=edge_attr)
                else:
                    out = conv(out, edge_index)
            if i != len(conv_layers) - 1:
                out = activation(out)

            # if FLAGS.ptrans and FLAGS.early_late_fusion == 'early':
            #     for tstype in self.seen_pragma_trans_types:
            #         if FLAGS.MLPs_share_weights:
            #             ts_mlp = self.tstype_MLPs[tstype]
            #         else:
            #             ts_mlp = self.tstype_MLPs[f'{tstype}_{i}']
            #         out = self._ptrans_apply_MLP(out, ts_mlp, data, tstype)

            outs.append(out)

        if FLAGS.jkn_enable:
            out = jkn(outs)
        return out, outs

    def node_att_gate_nn(self, D):
        if FLAGS.node_attention_MLP:
            return MLP(
                D,
                1,
                activation_type=FLAGS.activation_type,
                hidden_channels=[D // 2, D // 4, D // 8],
                num_hidden_lyr=3,
            )
        else:
            return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    def _gen_graph_level_embedding_from_node(
        self, node_emb, batch, data, out_dict, glob_P, glob_T, glob
    ):
        out = node_emb
        out_T = None
        out_P = None
        if FLAGS.node_attention and (glob_P is not None or glob_T is not None):
            out_gnn = out
            out_g = None
            out_P, out_T = None, None
            if FLAGS.separate_P and (glob_P is not None):
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = glob_P(out_gnn, batch)
                else:
                    out_P, node_att_scores_P = glob_P(
                        out_gnn, batch, set_zeros_ids=data.X_contextnids
                    )

                out_dict['emb_P'] = out_P
                out_g = out_P

            if FLAGS.separate_T and (glob_T is not None):
                out_T, node_att_scores = glob_T(
                    out_gnn, batch, set_zeros_ids=data.X_pragmanids
                )
                out_dict['emb_T'] = out_T
                if out_P is not None:
                    out_g = torch.cat((out_P, out_T), dim=1)
                else:
                    out_g = out_T

            if FLAGS.separate_pseudo and (self.glob_pseudo_B is not None):
                out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(
                    out_gnn, batch, set_zeros_ids=data.X_pseudonids
                )
                out_dict['emb_pseudo_b'] = out_pseudo_B
                if out_g is not None:
                    out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                else:
                    out_g = out_pseudo_B

            if FLAGS.separate_icmp and (self.glob_icmp is not None):
                out_icmp, node_att_scores_icmp = self.glob_icmp(
                    out_gnn, batch, set_zeros_ids=data.X_icmpnids
                )
                out_dict['emb_icmp'] = out_icmp
                if out_g is not None:
                    out_g = torch.cat((out_g, out_icmp), dim=1)
                else:
                    out_g = out_icmp

            if (
                not FLAGS.separate_P
                and not FLAGS.separate_T
                and not FLAGS.separate_pseudo
                and (glob_T is not None)
            ):
                out_g, node_att_scores = glob_T(out_gnn, batch)
                out_dict['emb_T'] = out
                if FLAGS.subtask == 'visualize':
                    # from saver import saver

                    saver.save_dict(
                        {'data': data, 'node_att_scores': node_att_scores},
                        f'node_att.pickle',
                    )

            out = out_g

        elif FLAGS.node_attention and glob is not None:
            out, node_att_scores = glob(node_emb, batch)
            out_T = out
        else:
            out = global_add_pool(out, batch)

            out_dict['emb_T'] = out
        if out_T is None:
            out_T = out
        return out_dict, out, out_P, out_T

        """ #commented by zongyue
        out_P = None
        if FLAGS.node_attention:
            if FLAGS.separate_P_T and glob_P is not None and glob_T is not None:
                # print(data.gname)'
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = glob_P(node_emb, batch)
                else:
                    out_P, node_att_scores_P = glob_P(node_emb, batch, set_zeros_ids=data.X_contextnids)

                out_T, node_att_scores = glob_T(node_emb, batch, set_zeros_ids=data.X_pragmanids)
                '''
                All of the following do not work due to the inplace modification not allowed 
                (would lead to RuntimeError).
                # node_emb[torch.nonzero(data.X_pragmanids)] = node_emb[torch.nonzero(data.X_pragmanids)].clone() + 5
                # node_emb[2] = node_emb[2] * node_emb[3].clone()
                # node_emb.index_add_(0, torch.tensor([0, 2, 3]), torch.ones(3, 64))
                # node_emb += 1         
                Therefore, to apply MLP/transformation to only certain nodes/rows, need to apply the MLP to all rows
                and then apply some masking.
                '''

                out_dict['emb_P'] = out_P

                out_embed = torch.cat((out_P, out_T), dim=1)

            else:
                out_T, node_att_scores = glob(node_emb, batch)
                # if FLAGS.subtask == 'visualize' and FLAGS.vis_what == 'att':
                #     saver.save_dict({'data': data, 'node_att_scores': node_att_scores},
                #                     f'{tvt}_{epoch}_{iter}_node_att.pickle')
                out_embed = out_T
        else:
            out_T = global_add_pool(node_emb, batch)
            out_embed = out_T
        out_dict['emb_T'] = out_T
        return out_dict, out_embed, out_P, out_T
        """

    # def _apply_target_MLPs_with_loss(
    #     self, mlps, out_embed, data, out_dict, loss_dict, mode
    # ):
    #     loss_rtn = 0.0
    #     if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
    #         out_MLPs = mlps(out_embed)
    #     for target_name in self.target_list:
    #         if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
    #             out = out_MLPs[target_name]
    #         else:
    #             out = mlps[target_name](out_embed)

    #         y = get_y_with_target(data, target_name)
    #         if self.task == 'regression':
    #             target = y.view((len(y), self.out_dim))
    #         else:
    #             target = y.view((len(y)))

    #         if mode == 'normal':
    #             # if self.task == 'regression' and FLAGS.loss_func == 'mse_weighted_util':
    #             #     loss = self.loss_function(out, target, target_name=target_name)
    #             # else:
    #             loss = self.loss_function(out, target)
    #             if (
    #                 hasattr(data, 'weight_switch')
    #                 and data.weight_switch == True
    #                 and 'util' in target_name
    #             ):
    #                 # weight = torch.clamp(1 - torch.sigmoid(-y+0.8), min=0.1)
    #                 weight = torch.clamp(
    #                     1 - torch.sigmoid(-out + 0.8), min=0.001
    #                 ).detach()
    #                 loss = weight * loss
    #             loss = loss.mean()
    #             target_name_s = target_name
    #         elif mode == 'pairwise_class':
    #             y1, y2 = self._split_vec_mat_into_2_halves(target)
    #             target = get_comp_result_tagret(y1, y2)
    #             assert out.shape[0] == target.shape[0]
    #             # saver.log_info(f'out={out}')
    #             # saver.log_info(f'target={target}')
    #             # saver.log_info(f'out.shape={out.shape}')
    #             # saver.log_info(f'target.shape={target.shape}')
    #             # saver.log_info(f'self.loss_function_pairwise_class={self.loss_function_pairwise_class}')
    #             loss = self.loss_function_pairwise_class(out, target)
    #             target_name_s = f'{target_name}_pairwise_class'
    #         else:
    #             assert False

    #         if FLAGS.loss_scale is not None:
    #             loss = loss * FLAGS.loss_scale[target_name]

    #         if FLAGS.margin_loss:
    #             sorted_out = out[torch.argsort(target, dim=0)].view(out.shape)
    #             shifted_delta = (sorted_out - torch.roll(sorted_out, -1, 0))[0:-1]
    #             margin_loss = torch.mean(
    #                 torch.max(
    #                     torch.zeros(shifted_delta.shape),  # .to(FLAGS.device),
    #                     shifted_delta,
    #                 )
    #             )
    #             print('margin loss', margin_loss)
    #             loss_rtn += margin_loss
    #         out_dict[target_name_s] = out

    #         use_regression_loss = True
    #         if (
    #             FLAGS.pairwise_class
    #             and 'pariwise_comparison' in FLAGS.pairwise_what_branches
    #             and FLAGS.loss_components == 'class_only'
    #         ):
    #             use_regression_loss = False

    #         if use_regression_loss:
    #             loss_rtn += loss
    #         loss_dict[target_name_s] = loss
    #         # print(target_name, 'loss', loss)
    #     return loss_rtn

    def _create_pragma_as_MLP_modules(self, D):
        # pragma as MLP
        pragma_as_MLP_list = FLAGS.pragma_as_MLP_list
        MLPs_per_pragma = nn.ModuleDict()
        for target in pragma_as_MLP_list:
            in_D = D + 1
            if target == 'parallel':
                in_D = D + 2  # reduction/normal, factor
            hidden_channels, len_hidden_channels = None, 0
            if FLAGS.pragma_MLP_hidden_channels is not None:
                hidden_channels = eval(FLAGS.pragma_MLP_hidden_channels)
                len_hidden_channels = len(hidden_channels)
            MLPs_per_pragma[target] = MLP(
                in_D,
                D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                num_hidden_lyr=len_hidden_channels,
            )
        if FLAGS.pragma_order == 'parallel_and_merge':
            in_D = D * len(pragma_as_MLP_list)
            hidden_channels = eval(FLAGS.merge_MLP_hidden_channels)

            MLPs_per_pragma['merge'] = MLP(
                in_D,
                D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                num_hidden_lyr=len(hidden_channels),
            )
        return pragma_as_MLP_list, MLPs_per_pragma

    def _apply_MLP_pragmas(
        self,
        out,
        data,
        edge_index,
        edge_attr,
        activation,
        conv_layers,
        pragma_as_MLP_list,
        MLPs_per_pragma,
    ):
        in_merge = None
        for pragma in pragma_as_MLP_list:
            out_MLP = self._apply_MLP_pragmas_helper(
                MLPs_per_pragma[pragma],
                out,
                data.X_pragmascopenids,
                data.X_pragma_per_node,
                pragma,
            )
            if FLAGS.pragma_order == 'sequential':
                out = out_MLP
            elif FLAGS.pragma_order == 'parallel_and_merge':
                if in_merge is None:
                    in_merge = out_MLP
                else:
                    in_merge = torch.cat((in_merge, out_MLP), dim=1)
            else:
                raise NotImplementedError()
        # the merge part
        if FLAGS.pragma_order == 'parallel_and_merge':
            out = self._apply_MLP_pragmas_helper(
                MLPs_per_pragma['merge'], out, data.X_pragmascopenids, in_merge, 'merge'
            )

        num_layers_for_MLP_pragma = FLAGS.num_conv_layers_for_MLP_pragma
        assert 0 <= num_layers_for_MLP_pragma <= len(conv_layers)
        index = len(conv_layers) - num_layers_for_MLP_pragma
        layers = conv_layers[index:]
        assert len(layers) == num_layers_for_MLP_pragma
        for i, conv in enumerate(layers):
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                out = conv(out, edge_index, edge_attr=edge_attr)
            else:
                out = conv(out, edge_index)
            if i != len(conv_layers) - 1:
                out = activation(out)
        return out

    def _apply_MLP_pragmas_helper(
        self, mlp_pragma, out, scope_nodes, X_pragma_per_node, ptype
    ):
        mlp_inp, pragma_option = None, None
        if ptype == 'tile':
            pragma_option = X_pragma_per_node[:, 0].reshape(-1, 1)
        elif ptype == 'pipeline':
            pragma_option = X_pragma_per_node[:, 1].reshape(-1, 1)
        elif ptype == 'parallel':
            pragma_option = X_pragma_per_node[:, 2:4].reshape(-1, 2)
        elif ptype == 'merge':
            mlp_inp = X_pragma_per_node
        else:
            raise NotImplementedError()

        non_scope_nodes = torch.sub(1, scope_nodes)
        # masked_emb = scope_nodes.ge(0.5)
        if ptype == 'merge':
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self._mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            out = self._mask_emb(out, non_zero_ids=non_scope_nodes) + self._mask_emb(
                mlp_out, non_zero_ids=scope_nodes
            )
        else:
            mlp_inp = torch.cat((out, pragma_option), dim=1)
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out = torch.clone(out)
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self._mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            if FLAGS.pragma_order == 'sequential':
                out = self._mask_emb(
                    out, non_zero_ids=non_scope_nodes
                ) + self._mask_emb(mlp_out, non_zero_ids=scope_nodes)
            elif FLAGS.pragma_order == 'parallel_and_merge':
                out = self._mask_emb(mlp_out, non_zero_ids=scope_nodes)
            else:
                raise NotImplementedError()

        return out

    def _mask_emb(self, out, non_zero_ids):
        out = out.permute((1, 0))
        out = out * non_zero_ids
        out = out.permute((1, 0))

        return out


class Net(Model):
    def __init__(self, task, init_pragma_dict=None, dataset=None, D=None):
        super(Net, self).__init__()
        self.MLP_version = 'multi_obj' if len(FLAGS.target) > 1 else 'single_obj'

        in_channels = get_num_features(dataset)
        if FLAGS.encode_edge:
            edge_attr = dataset[0].edge_attr
            if edge_attr is None:
                edge_attr = dataset[0].edge_attr_programl
            edge_dim = edge_attr.shape[1]
        else:
            edge_dim = -1

        num_layers = FLAGS.num_layers
        D = FLAGS.D
        # target = FLAGS.target
        if dataset is not None:
            self.num_features_node = dataset.num_features

        if FLAGS.sequence_modeling:
            self.bert_model, self.bert_dim = self._create_src_code_bert()

        conv_class = self._get_GNN_conv_class()

        self.conv_first = self._create_conv_first(
            edge_dim, in_channels, D, conv_class, FLAGS.sequence_modeling
        )

        self.conv_layers, self.jkn = self._create_GNN_conv_layers(
            num_layers, edge_dim, D, conv_class
        )

        if FLAGS.pragma_as_MLP:
            self.pragma_as_MLP_list, self.MLPs_per_pragma = (
                self._create_pragma_as_MLP_modules(D)
            )

        # if FLAGS.ptrans:
        #     self.tstype_MLPs = self._create_ptrans_MLPs(dataset, D)

        if FLAGS.node_attention:
            if FLAGS.separate_P_T:
                self.gate_nn_T, self.glob_T = create_graph_att_module(
                    D, return_gate_nn=True
                )
                if FLAGS.separate_P_T:
                    self.gate_nn_P, self.glob_P = create_graph_att_module(
                        D, return_gate_nn=True
                    )
                # if FLAGS.separate_pseudo:  ## for now, only pseudo node for block
                #     self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                #     self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
            else:
                self.gate_nn, self.glob = create_graph_att_module(
                    D, return_gate_nn=True
                )

        if FLAGS.gae_T:  # graph auto encoder
            if FLAGS.separate_P_T:
                self.gae_transform_T = nn.ModuleDict()
                for gname, feat_dim in init_pragma_dict.items():
                    mlp = Linear(feat_dim[0], D // 8)
                    if FLAGS.pragma_uniform_encoder:
                        self.gae_transform_T['all'] = Linear(feat_dim[1], D // 8)
                        break
                    else:
                        self.gae_transform_T[gname] = mlp
                channels = [D // 2, D // 4]
                self.decoder_T = MLP(
                    D,
                    D // 8,
                    activation_type=FLAGS.activation,
                    hidden_channels=channels,
                    num_hidden_lyr=len(channels),
                )
        if FLAGS.gae_P:
            out_channels = in_channels
            if FLAGS.input_encode:
                # encode input one-hot representation
                self.gate_input = Linear(in_channels, 2 * D)
                out_channels = 2 * D

            if FLAGS.decoder_type == 'type1':
                decoder_arch = []
            elif FLAGS.decoder_type == 'type2':
                decoder_arch = [D, 2 * D, out_channels]
            self.decoder_P = MLP(
                D,
                out_channels,
                activation_type=FLAGS.activation,
                hidden_channels=decoder_arch,
                num_hidden_lyr=len(decoder_arch),
            )
            if FLAGS.decoder_type == 'None':
                for name, param in self.decoder_P.named_parameters():
                    print(name)
                    param.requires_grad = False
        if FLAGS.gae_T or FLAGS.gae_P:
            self.gae_sim_function = nn.CosineSimilarity()
            self.gae_loss_function = nn.CosineEmbeddingLoss()

        self.task = task

        self.out_dim, self.loss_function = self._create_loss()

        self.target_list = self._get_target_list()

        if FLAGS.pairwise_class:
            self.loss_function_pairwise_class = nn.CrossEntropyLoss()

            if 'pariwise_comparison' in FLAGS.pairwise_what_branches and 'node_att_diff' in FLAGS.comp_ops:
                self.node_att_diff = NodeAttDiff(node_dim=D)


        if FLAGS.node_attention:
            if FLAGS.separate_P_T:
                in_D = 2 * D
            else:
                in_D = D
        else:
            in_D = D
        self.MLPs, self.MLPs_pairwise_class = self._create_decoder_MLPs(
            in_D, D, self.target_list, self.out_dim
        )

        has_guidance = FLAGS.load_pretrained_GNN
        if hasattr(FLAGS, "load_guidance_emb") and FLAGS.load_guidance_emb == True:
            has_guidance = True
        if has_guidance and FLAGS.D != 64:
            self.node_embs_proj_to_pretrained_model = MLP(
                D,
                64,
                activation_type=FLAGS.activation_type,
                num_hidden_lyr=4,
                hidden_channels=[int(D * 0.75), int(D * 0.5), int(D * 0.25), 64],
            )

    # def _create_ptrans_MLPs(self, dataset, D):
    #     if dataset is None:
    #         raise RuntimeError(f'If ptrans is set to True, must send dataset to model init')
    #     tstype_MLPs = nn.ModuleDict()
    #     self.seen_pragma_trans_types = dataset.get_attribute('seen_pragma_trans_types')
    #     for tstype in self.seen_pragma_trans_types:
    #         if FLAGS.early_late_fusion == 'early':
    #             if not FLAGS.MLPs_share_weights:
    #                 for i, _ in enumerate(self.conv_layers):
    #                     tstype_MLPs[f'{tstype}_{i}'] = MLP(D, D, activation_type=FLAGS.activation)
    #             else:
    #                 tstype_MLPs[tstype] = MLP(D, D, activation_type=FLAGS.activation)
    #         else:
    #             assert FLAGS.early_late_fusion == 'late'
    #             tstype_MLPs[tstype] = MLP(D, D, activation_type=FLAGS.activation)
    #     return tstype_MLPs

    def cal_gae_loss(self, encoded_g, decoded_out):
        target = torch.ones(
            len(encoded_g), device=decoded_out.device
        )  # for similarity, use the negative form for dissimilarity
        target.requires_grad = False
        gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        return gae_loss

    def forward(
        self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None
    ):
        if FLAGS.graph_transformer_option is not None:
            raise RuntimeError(f'Deprecated!')
            gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
            if gt_conv_type == 'mha':
                x = data.X_padded
            elif gt_conv_type == 'gps_conv':
                x = data.x
            else:
                raise NotImplementedError()
            edge_index, edge_attr, batch, pragmas = (
                data.edge_index,
                getattr(data, 'edge_attr'),
                data.batch,
                getattr(data, 'pragmas', None),
            )
        else:
            x, edge_index, edge_attr, batch, pragmas = (
                data.x,
                data.edge_index,
                getattr(data, 'edge_attr'),
                data.batch,
                getattr(data, 'pragmas', None),
            )

            if x is None:

                assert not FLAGS.multi_modality
                if FLAGS.sequence_modeling:
                    x = data.x_src_code
                    edge_index = data.edge_index_src_code
                    batch = data.x_src_code_batch
                else:
                    x = data.x_programl
                    edge_index = data.edge_index_programl
                    edge_attr = data.edge_attr_programl
                    batch = data.x_programl_batch

                if x is None or edge_index is None or batch is None:
                    raise RuntimeError(f'x={x}; edge_index={edge_index}; batch={batch}')

            num_features_node = getattr(self, 'num_features_node')
            if (
                num_features_node is not None
                and num_features_node != 0
                and hasattr(data, 'xy_dict_programl')
            ):
                if x.shape[1] != num_features_node:
                    saver.log_info(
                        f'Weird error; x.shape={x.shape} while num_features_node={self.num_features_node};'
                    )
                    for gname, point in zip(data.gname, data.xy_dict_programl['point']):
                        saver.log_info(f'\t{gname}: {point}')
                    raise RuntimeError(f'Bad input batch; need to debug!')

        gname = None
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        # print(gname)
        # print(edge_attr.shape)
        outs = []
        out_dict = OrderedDict()

        activation = self._get_act_func()

        if FLAGS.sequence_modeling:
            x_bert, _ = self._apply_bert(x, data)
            x = x_bert

            if FLAGS.data_repr == 'ast' and FLAGS.AST_combine_node_edge_labels:
                x = torch.cat((x_bert, data.X_ast_node_labels), dim=1)

        if FLAGS.sequence_modeling and not FLAGS.apply_act_conv_first:
            act = None
        else:
            act = activation
        out = self._apply_conv_first(self.conv_first, x, edge_index, edge_attr, act)

        outs.append(out)

        out, outs = self._apply_conv_layers(
            out,
            self.conv_layers,
            self.jkn,
            outs,
            edge_index,
            edge_attr,
            batch,
            data,
            activation,
        )

        if FLAGS.pragma_as_MLP:
            out = self._apply_MLP_pragmas(
                out,
                data,
                edge_index,
                edge_attr,
                activation,
                self.conv_layers,
                self.pragma_as_MLP_list,
                self.MLPs_per_pragma,
            )

        # if FLAGS.ptrans and FLAGS.early_late_fusion == 'late':
        #     for tstype in self.seen_pragma_trans_types:
        #         ts_mlp = self.tstype_MLPs[tstype]
        #         out = self._ptrans_apply_MLP(out, ts_mlp, data, tstype)

        if FLAGS.graph_transformer_option is not None:
            gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
            if gt_conv_type == 'mha':
                out = torch.nn.utils.rnn.pack_padded_sequence(
                    out,
                    lengths=data.X_padded_lengths,
                    batch_first=True,
                    enforce_sorted=False,
                ).data
                assert out.shape[0] == data.x.shape[0]
            elif gt_conv_type == 'gps_conv':
                pass
            else:
                raise NotImplementedError()

        node_emb = out
        total_loss = 0.0

        if FLAGS.load_pretrained_GNN and FLAGS.D != 64:
            out_dict['node_emb'] = self.node_embs_proj_to_pretrained_model(node_emb)
        elif (
            hasattr(FLAGS, "load_guidance_emb")
            and FLAGS.load_guidance_emb
            and FLAGS.D != 64
        ):
            out_dict['node_emb'] = self.node_embs_proj_to_pretrained_model(node_emb)
        else:
            out_dict['node_emb'] = node_emb

        out_dict, out_embed, out_P, out_T = self._gen_graph_level_embedding_from_node(
            node_emb,
            batch,
            data,
            out_dict,
            glob_P=getattr(self, 'glob_P', None),
            glob_T=getattr(self, 'glob_T', None),
            glob=getattr(self, 'glob', None),
        )

        # out = global_add_pool(out, batch)
        # out, edge_index, _, batch, perm, score = self.pool1(
        #     out, edge_index, None, batch)
        # ratio = out.size(0) / x.size(0)

        gae_loss = torch.tensor(0.0, device=out_embed.device)
        if FLAGS.gae_T:  # graph auto encoder
            if FLAGS.separate_P_T:
                if FLAGS.pragma_uniform_encoder:
                    gname = 'all'
                encoded_g = self.gae_transform_T[gname](pragmas)
                decoded_out = self.decoder_T(out_dict['emb_T'])
                # gae_loss = self.cal_gae_loss(encoded_g, decoded_out)
                target = torch.ones(
                    len(encoded_g), device=decoded_out.device
                )  # for similarity, use the negative form for dissimilarity
                target.requires_grad = False
                gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        if FLAGS.gae_P:
            encoded_x = x
            if FLAGS.input_encode:
                encoded_x = self.gate_input(x)
            # simple addition of node embeddings for gae
            encoded_g = global_add_pool(encoded_x, batch)

            if FLAGS.decoder_type == 'None':  # turn off autograd:
                decoded_out = self.decoder_P(out_dict['emb_P']).detach()
            else:
                decoded_out = self.decoder_P(out_dict['emb_P'])
            # gae_loss = (self.gae_loss_function(encoded_g, decoded_out)).mean()
            gae_loss += self.cal_gae_loss(encoded_g, decoded_out)
            # from saver import saver
            # saver.info(f'cosine similarity is {self.gae_sim_function(encoded_g, decoded_out).mean()}')
            # saver.log_info(f'encoded_g : {F.normalize(encoded_g[0, :], dim=0)}')
            # saver.log_info(f'decoded_out : {F.normalize(decoded_out[0, :], dim=0)}')
        if FLAGS.gae_P or FLAGS.gae_T:
            total_loss += torch.abs(gae_loss)
            # gae_loss = gae_loss.view((len(gae_loss), 1))
            # print(gae_loss.shape)

        loss_dict = OrderedDict()

        if FLAGS.subtask == 'inference' and FLAGS.save_emb:
            d = {'data.gname': data.gname, 'data.key': data.key, 'out_embed': out_embed}
            for target_name in self.target_list:
                d[target_name] = get_y_with_target(data, target_name)
            saver.save_emb_accumulate_emb(
                f'{data.gname}_{data.key}', d, convert_to_np=True
            )

        if forward_pairwise:
            strict_even = True
        else:
            strict_even = False
            
        total_loss += self._apply_target_MLPs_with_loss(
            self.MLPs, out_embed, data, out_dict, loss_dict, 'normal', strict_even
        )

        if (
            forward_pairwise
            and FLAGS.pairwise_class
            and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            and FLAGS.loss_components in ['both', 'class_only']
        ):
            pairwise_comp_results = []
            d1_gemb, d2_gemb = split_vec_mat_into_2_halves(out_embed)
            for op in FLAGS.comp_ops:
                if op == 'hadamard':
                    c = d1_gemb * d2_gemb
                elif op == 'diff':
                    c = d1_gemb - d2_gemb
                elif op == 'emb_d1':
                    c = d1_gemb
                elif op == 'emb_d2':
                    c = d2_gemb
                elif op == 'node_att_diff':
                    c = self.node_att_diff(node_emb, batch)
                else:
                    raise NotImplementedError()
                pairwise_comp_results.append(c)
            pairwise_comp_results = torch.cat(pairwise_comp_results, dim=1)





            total_loss += self._apply_target_MLPs_with_loss(
                self.MLPs_pairwise_class,
                pairwise_comp_results,
                data,
                out_dict,
                loss_dict,
                'pairwise_class',
                strict_even
            )
            
        if FLAGS.itype_mask_perc > 0:
            y = self.mask_MLP_vocab(node_emb)
            y = self.mask_softmax(y)
            mask_loss = self.mask_loss(y, data['itype_true_labels'])
            total_loss += mask_loss
            loss_dict['itype_mask'] = mask_loss

        return out_dict, total_loss, loss_dict, gae_loss

    def _apply_target_MLPs_with_loss(
        self, mlps, out_embed, data, out_dict, loss_dict, mode, strict_even
    ):
        loss_rtn = 0.0
        if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
            out_MLPs = mlps(out_embed)
        for target_name in self.target_list:
            if mode == 'normal':
                target_name_s = target_name
            elif mode == 'pairwise_class':
                target_name_s = f'{target_name}_pairwise_class'
            else:
                assert False

            if FLAGS.loss_components_to_keep != 'all' and target_name_s not in FLAGS.loss_components_to_keep and mode == 'pairwise_class':
                saver.log_info_once(f'Ignore loss component {target_name_s}')
                continue

            if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
                out = out_MLPs[target_name]
            else:
                out = mlps[target_name](out_embed)

            out_dict[target_name_s] = out

            if FLAGS.loss_components_to_keep != 'all' and target_name_s not in FLAGS.loss_components_to_keep:
                # saver.log_info_once(f'Ignore loss component {target_name_s}')
                continue


            y = get_y_with_target(data, target_name)
            if self.task == 'regression':
                target = y.view((len(y), self.out_dim))
            else:
                target = y.view((len(y)))

            if mode == 'normal':
                # if self.task == 'regression' and FLAGS.loss_func == 'mse_weighted_util':
                #     loss = self.loss_function(out, target, target_name=target_name)
                # else:
                loss = self.loss_function(out, target)
                if (
                    hasattr(data, 'weight_switch')
                    and data.weight_switch == True
                    and 'util' in target_name
                ):
                    # weight = torch.clamp(1 - torch.sigmoid(-y+0.8), min=0.1)
                    weight = torch.clamp(
                        1 - torch.sigmoid(-out + 0.8), min=0.001
                    ).detach()
                    loss = weight * loss
                loss = loss.mean()
            elif mode == 'pairwise_class':
                y1, y2 = split_vec_mat_into_2_halves(target, strict_even=strict_even)
                target_pairwise = get_comp_result_tagret(y1, y2)
                assert out.shape[0] == target_pairwise.shape[0]
                loss = self.loss_function_pairwise_class(out, target_pairwise)
                # loss = self.loss_function_pairwise_class(out, target, strict_even=strict_even)

            else:
                assert False

            if FLAGS.loss_scale is not None:
                loss = loss * FLAGS.loss_scale[target_name]

            if FLAGS.margin_loss:
                sorted_out = out[torch.argsort(target, dim=0)].view(out.shape)
                shifted_delta = (sorted_out - torch.roll(sorted_out, -1, 0))[0:-1]
                margin_loss = torch.mean(
                    torch.max(
                        torch.zeros(shifted_delta.shape),  # .to(FLAGS.device),
                        shifted_delta,
                    )
                )
                print('margin loss', margin_loss)
                loss_rtn += margin_loss
            # out_dict[target_name_s] = out

            # use_regression_loss = True
            # if (
            #     FLAGS.pairwise_class
            #     and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            #     and FLAGS.loss_components in ['both', 'class_only']
            # ):
            #     use_regression_loss = False

            # if use_regression_loss:
            loss_rtn += loss
            loss_dict[target_name_s] = loss
            # print(target_name, 'loss', loss)


            if mode == 'pairwise_class':
                if FLAGS.branch_discrepancy_loss:
                    y1_pred, y2_pred = split_vec_mat_into_2_halves(out_dict[target_name], strict_even=strict_even)
                    f_point = torch.cat([y1_pred, y2_pred], dim=1)
                    f_comp = out_dict[target_name_s]
                    saver.log_info_at_most(f'f_point={f_point}', f'bcl_fp', 1)
                    saver.log_info_at_most(f'f_comp={f_comp}', f'bcl_fc', 1)
                    loss = self._branch_discrepancy_loss(f_point, f_comp)
                    loss_dict[f'{target_name_s}_branch_discrepancy'] = loss
                    loss_rtn += loss
                    saver.log_info_at_most(f'loss_dict={loss_dict}', f'bcl_ld', 1)

            if FLAGS.listmle_loss: # applied to the pointwise branch
                predicted = out_dict[target_name]
                true = target
                saver.log_info_at_most(f'predicted={predicted}', f'lmle_p', 5)
                saver.log_info_at_most(f'true={true}', f'lmle_t', 5)
                loss = self._listMLE_loss(predicted, true)
                loss_dict[f'{target_name_s}_listmle'] = loss
                loss_rtn += loss
                saver.log_info_at_most(f'ListMLE loss={loss}', f'lmle_p', 5)

                    # exit()
        return loss_rtn
    
    def _branch_discrepancy_loss(self, f_point, f_comp):
        # f_point: [batch_size, 2] (pointwise predictions for two designs in each pair)
        # f_comp: [batch_size, 2] (logits for pairwise comparisons)
        
        # Calculate the absolute difference of pointwise predictions
        delta = torch.abs(f_point[:, 0] - f_point[:, 1])  # [batch_size]

        # Calculate softmax probabilities from pairwise logits
        softmax_prob = F.softmax(f_comp, dim=1)  # [batch_size, 2]

        # Determine the wrong class based on pointwise predictions
        wrong_class = (f_point[:, 0] > f_point[:, 1]).long()  # [batch_size]
        
        # Discrepancy loss
        discrepancy_loss = softmax_prob[torch.arange(softmax_prob.size(0)), wrong_class] * delta

        # Hyperparameters to balance the loss components
        alpha = 1.0

        # Combined loss
        total_loss = alpha * torch.mean(discrepancy_loss)# + beta * torch.mean(consistency_reward)

        return total_loss
    
    def _listMLE_loss(self, predicted, true):
        """
        Computes ListMLE loss for given predictions and true values.
        Args:
        predicted (torch.Tensor): Predicted values from the model.
        true (torch.Tensor): True values corresponding to the predictions.

        Returns:
        float: Computed ListMLE loss.
        """
        _, sorted_indices = torch.sort(true, descending=True)
        predicted_sorted = predicted.gather(0, sorted_indices)
        # return torch.sum(torch.logcumsumexp(-predicted_sorted, dim=0)) - torch.sum(-predicted_sorted)
        log_cumsum = torch.logcumsumexp(-predicted_sorted, dim=0)
        loss = torch.sum(log_cumsum) - torch.sum(-predicted_sorted)
        normalized_loss = loss / predicted_sorted.size(0)  # Normalizing by the length of the list
        return normalized_loss


    def _create_target_MLPs_dict(self, target_list, hidden_channels, in_D, out_D):

        if (not hasattr(self, "MLP_version")) or self.MLP_version == 'single_obj':
            MLPs = nn.ModuleDict()
            for target in self.target_list:
                MLPs[target] = MLP(
                    in_D,
                    out_D,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    num_hidden_lyr=len(hidden_channels),
                )
        else:
            MLPs = MLP_multi_objective(
                in_D,
                out_D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                objectives=self.target_list,
                num_common_lyr=FLAGS.MLP_common_lyr,
            )

        """ #commeted by zongyue
        rtn = nn.ModuleDict()
        
        for target in target_list:
            rtn[target] = MLP(in_D, out_D, activation_type=FLAGS.activation,
                              hidden_channels=hidden_channels,
                              num_hidden_lyr=len(hidden_channels))
        return rtn
        """
        return MLPs


class MAPE(torch.nn.Module):
    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert (
                    y_pred.size(-1) == 1
                ), "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def forward(self, y_pred, target):
        a = (self.to_prediction(y_pred) - target).abs()
        b = target.abs() + 1e-8
        loss = a / b
        rtn = torch.mean(loss)
        return rtn


class MSE_WEIGHT_UTIL(torch.nn.Module):
    def forward(self, y_pred, target, target_name):
        loss = (y_pred - target) ** 2
        if 'util' in target_name:
            loss = loss * torch.exp(y_pred - 1)
        rtn = torch.mean(loss)
        return rtn


def get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))


def _create_gnn(D1, D2):
    if FLAGS.gnn_type == 'gat':
        conv_class = GATConv
    elif FLAGS.gnn_type == 'gcn':
        conv_class = GCNConv
    elif FLAGS.gnn_type == 'transformer':
        conv_class = TransformerConv
    else:
        raise NotImplementedError()

    if FLAGS.gnn_type == 'gcn':
        return conv_class(D1, D2)
    else:
        return conv_class(D1, D2, heads=1)


def feature_extract(model, key_word, gnn_layer=None):
    '''"
    fixes all parameters except for the ones that have "key_word"
    as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                saver.log_info(f'fixing parameter: {name}')
                param.requires_grad = False
            else:
                if 'conv_first' in name or any(
                    [f'conv_layers.{d}' in name for d in range(gnn_layer - 1)]
                ):
                    saver.log_info(f'fixing parameter: {name}')
                    param.requires_grad = False

    if FLAGS.random_MLP:
        D = FLAGS.D
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
        for target in FLAGS.target:
            model.MLPs[target] = MLP(
                D,
                1,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                num_hidden_lyr=len(hidden_channels),
            )


def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
    checks that all parameters except for the ones that have "key_word" are fixed
    as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any(
                    [f'conv_layers.{d}' in name for d in range(gnn_layer - 1)]
                ):
                    assert param.requires_grad == False


class HARPNet(nn.Module):
    def __init__(
        self,
        in_channels,
        edge_dim=0,
        init_pragma_dict=None,
        task=FLAGS.task,
        num_layers=FLAGS.num_layers,
        D=FLAGS.D,
        target=FLAGS.target,
    ):
        super(HARPNet, self).__init__()

        # Specially written for Atefeh's model.
        self.separate_P = True
        self.separate_pseudo = True
        SSL = False
        self.pragma_as_MLP = True
        self.pragma_as_MLP_list = ['tile', 'pipeline', 'parallel']
        self.pragma_MLP_hidden_channels = '[in_D // 2]'
        self.merge_MLP_hidden_channels = '[in_D // 2]'
        self.pragma_order = 'parallel_and_merge'

        # saver.log_info(f'HARPNet: in_channels={in_channels}')
        self.MLP_version = 'multi_obj' if len(FLAGS.target) > 1 else 'single_obj'
        conv_class = TransformerConv

        self.conv_first = conv_class(
            in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout
        )

        self.conv_layers = nn.ModuleList()

        self.num_conv_layers = num_layers - 1
        num_layers += FLAGS.gnn_layer_after_MLP
        for _ in range(num_layers - 1):
            conv = conv_class(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
            self.conv_layers.append(conv)

        self.jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)

        self.task = task

        if task == 'regression':
            if False:  # 'GNLL' in FLAGS.loss: 03/19/2024: Let's just do the regular way
                self.out_dim = 1
                self.MLP_out_dim = 2
                if FLAGS.loss == 'myGNLL':
                    self.loss_function = self.gaussianNLL
                else:
                    self.loss_function = nn.GaussianNLLLoss()
                    self.my_softplus = nn.Softplus()

            else:
                self.out_dim = 1
                self.MLP_out_dim = 1
                self.loss_function = nn.MSELoss()
            self.target_list = FLAGS.target

        else:
            self.out_dim = 2
            self.MLP_out_dim = 2
            self.loss_function = nn.CrossEntropyLoss()
            self.target_list = ['perf']

        if FLAGS.node_attention:
            if self.separate_P:
                self.gate_nn_P = self.node_att_gate_nn(D)
                self.glob_P = MyGlobalAttention(self.gate_nn_P, None)
            if self.separate_pseudo:  ## for now, only pseudo node for block
                self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)

        if not SSL:
            if FLAGS.node_attention:
                dim = self.separate_P + self.separate_pseudo
                in_D = dim * D
            else:
                in_D = D
            if D > 64:
                hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
            else:
                hidden_channels = [D // 2, D // 4, D // 8]
            if self.MLP_version == 'single_obj':
                self.MLPs = nn.ModuleDict()
                for target in self.target_list:
                    self.MLPs[target] = MLP(
                        in_D,
                        self.MLP_out_dim,
                        activation_type=FLAGS.activation,
                        hidden_channels=hidden_channels,
                        num_hidden_lyr=len(hidden_channels),
                    )
            else:
                self.MLPs = MLP_multi_objective(
                    in_D,
                    self.MLP_out_dim,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    objectives=self.target_list,
                    num_common_lyr=FLAGS.MLP_common_lyr,
                )

            if (
                FLAGS.pairwise_class
                and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            ):
                
                in_D_class = 0
                for op in FLAGS.comp_ops:
                    if op == 'node_att_diff':
                        in_D_class += D
                    else:
                        in_D_class += in_D
                # in_D_class = in_D * len(FLAGS.comp_ops)
                # TODO: choice: compare before each target or for/within/dedicated to each target

                # hidden_channels_pairwise = []
                # D_temp = in_D_class
                # while D_temp >= 1:
                #      hidden_channels_pairwise.append(D_temp // 2)
                #      D_temp = D_temp // 2
                # hidden_channels_pairwise = [in_D_class // 2**i for i in range(1, math.floor(math.log2(in_D_class)) + 1)]

                hidden_channels_pairwise =  deduce_MLP_dimensions(in_D_class, 2, reduce_factor=2)

                # saver.log_info(f'hidden_channels_pairwise={hidden_channels_pairwise}')


                self.MLPs_pairwise_class = self._create_target_MLPs_dict(
                    self.target_list, hidden_channels_pairwise, in_D_class, 2 # TODO: previously: hidden_channels
                )  # binary


                if 'node_att_diff' in FLAGS.comp_ops:
                    self.node_att_diff = NodeAttDiff(node_dim=D)


            if FLAGS.pairwise_class:
                # self.loss_function_pairwise_class = nn.CrossEntropyLoss()
                self.loss_function_pairwise_class = PairwiseLoss()

        ## pragma as MLP
        if self.pragma_as_MLP:
            self.pragma_as_MLP_list = self.pragma_as_MLP_list
            self.MLPs_per_pragma = nn.ModuleDict()
            for target in self.pragma_as_MLP_list:
                in_D = D + 1
                if target == 'parallel':
                    in_D = D + 2  ## reduction/normal, factor
                hidden_channels, len_hidden_channels = None, 0
                if self.pragma_MLP_hidden_channels is not None:
                    hidden_channels = eval(self.pragma_MLP_hidden_channels)
                    len_hidden_channels = len(hidden_channels)
                self.MLPs_per_pragma[target] = MLP(
                    in_D,
                    D,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    num_hidden_lyr=len_hidden_channels,
                )
            if self.pragma_order == 'parallel_and_merge':
                in_D = D * len(self.pragma_as_MLP_list)
                hidden_channels = eval(self.merge_MLP_hidden_channels)

                self.MLPs_per_pragma['merge'] = MLP(
                    in_D,
                    D,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    num_hidden_lyr=len(hidden_channels),
                )

    def node_att_gate_nn(self, D):
        if FLAGS.node_attention_MLP:
            return MLP(
                D,
                1,
                activation_type=FLAGS.activation_type,
                hidden_channels=[D // 2, D // 4, D // 8],
                num_hidden_lyr=3,
            )
        else:
            return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    def cal_gae_loss(self, encoded_g, decoded_out):
        target = torch.ones(
            len(encoded_g), device=encoded_g.device
        )  ## for similarity, use the negative form for dissimilarity
        target.requires_grad = False
        gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        return gae_loss

    def gaussianNLL(self, out=None, target=None):
        '''
        out should include mu and sigma
        https://towardsdatascience.com/get-uncertainty-estimates-in-neural-networks-for-free-48f2edb82c8f
        '''
        if out is not None:
            mu = out[:, 0].reshape(-1, 1)
            log_var = out[:, 1].reshape(-1, 1)
            # mu = torch.zeros(log_var.shape).to(FLAGS.device)
            var = torch.exp(log_var)

            if FLAGS.beta > 0:
                scalar = var.detach() ** FLAGS.beta
            else:
                scalar = torch.ones(var.shape).to(out.device)

            return torch.mean(
                (log_var / 2 + (1 / 2) * (1 / var) * ((target - mu)) ** 2) * scalar
            )
        else:
            return None

    def mask_emb(self, out, non_zero_ids):
        out = out.permute((1, 0))
        out = out * non_zero_ids
        out = out.permute((1, 0))

        return out

    def apply_pragam_as_MLP(
        self, mlp_pragma, out, scope_nodes, X_pragma_per_node, ptype
    ):
        if ptype == 'tile':
            pragma_option = X_pragma_per_node[:, 0].reshape(-1, 1)
        elif ptype == 'pipeline':
            pragma_option = X_pragma_per_node[:, 1].reshape(-1, 1)
        elif ptype == 'parallel':
            pragma_option = X_pragma_per_node[:, 2:4].reshape(-1, 2)
        elif ptype == 'merge':
            mlp_inp = X_pragma_per_node
        else:
            raise NotImplementedError()

        non_scope_nodes = torch.sub(1, scope_nodes)
        masked_emb = scope_nodes.ge(0.5)
        if ptype == 'merge':
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            out = self.mask_emb(out, non_zero_ids=non_scope_nodes) + self.mask_emb(
                mlp_out, non_zero_ids=scope_nodes
            )
        else:
            mlp_inp = torch.cat((out, pragma_option), dim=1)
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out = torch.clone(out)
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            if self.pragma_order == 'sequential':
                out = self.mask_emb(out, non_zero_ids=non_scope_nodes) + self.mask_emb(
                    mlp_out, non_zero_ids=scope_nodes
                )
            elif self.pragma_order == 'parallel_and_merge':
                out = self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
            else:
                raise NotImplementedError()

        return out

    def forward(
        self, data, forward_pairwise=None, tvt=None, iter=None, test_name=None, *_
    ):
        x, edge_index, edge_attr, batch, pragmas = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.pragmas,
        )
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        if hasattr(data, 'X_pragma_per_node'):
            X_pragma_per_node = data.X_pragma_per_node
        # print(gname)
        # print(edge_attr.shape)
        outs = []
        out_dict = OrderedDict()
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()

        out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))

        outs.append(out)

        # for i, conv in enumerate(self.conv_layers):
        for i in range(self.num_conv_layers):
            conv = self.conv_layers[i]
            out = conv(out, edge_index, edge_attr=edge_attr)
            if i != len(self.conv_layers) - 1:
                out = activation(out)

            outs.append(out)

        if FLAGS.jkn_enable:
            out = self.jkn(outs)

        ## pragma as MLP
        if self.pragma_as_MLP:
            in_merge = None
            for pragma in self.pragma_as_MLP_list:
                out_MLP = self.apply_pragam_as_MLP(
                    self.MLPs_per_pragma[pragma],
                    out,
                    data.X_pragmascopenids,
                    X_pragma_per_node,
                    pragma,
                )
                if self.pragma_order == 'sequential':
                    out = out_MLP
                elif self.pragma_order == 'parallel_and_merge':
                    if in_merge is None:
                        in_merge = out_MLP
                    else:
                        in_merge = torch.cat((in_merge, out_MLP), dim=1)
                else:
                    raise NotImplementedError()
            ## the merge part
            if self.pragma_order == 'parallel_and_merge':
                out = self.apply_pragam_as_MLP(
                    self.MLPs_per_pragma['merge'],
                    out,
                    data.X_pragmascopenids,
                    in_merge,
                    'merge',
                )

            for i, conv in enumerate(self.conv_layers[self.num_conv_layers :]):
                out = conv(out, edge_index, edge_attr=edge_attr)
                if i != len(self.conv_layers) - 1:
                    out = activation(out)

        if FLAGS.node_attention:
            out_gnn = out
            out_g = None
            out_P, out_T = None, None
            if self.separate_P:
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = self.glob_P(out_gnn, batch)
                else:
                    out_P, node_att_scores_P = self.glob_P(
                        out_gnn, batch, set_zeros_ids=data.X_contextnids
                    )

                out_dict['emb_P'] = out_P
                out_g = out_P

            if self.separate_pseudo:
                out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(
                    out_gnn, batch, set_zeros_ids=data.X_pseudonids
                )
                out_dict['emb_pseudo_b'] = out_pseudo_B
                if out_g is not None:
                    out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                else:
                    out_g = out_pseudo_B

            if not self.separate_P and not self.separate_pseudo:
                out_g, node_att_scores = self.glob_T(out_gnn, batch)
                out_dict['emb_T'] = out
                if FLAGS.subtask == 'visualize':
                    saver.save_dict(
                        {'data': data, 'node_att_scores': node_att_scores},
                        f'node_att.pickle',
                    )

            out = out_g
        else:
            out = global_add_pool(out, batch)
            out_dict['emb_T'] = out

        total_loss = 0
        gae_loss = torch.tensor(0.0, device=out.device)

        out_embed = out
        # _target_list = FLAGS.target
        # if not isinstance(FLAGS.target, list):
        #     _target_list = [FLAGS.target]
        # target_list = [t for t in _target_list]
        # target_list = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
        # target_list = ['util-DSP']

        loss_dict = {}
        if not FLAGS.SSL:
            if self.MLP_version == 'multi_obj':
                out_MLPs = self.MLPs(out_embed)
            for target_name in self.target_list:
                # for target_name in target_list:
                if self.MLP_version == 'multi_obj':
                    out = out_MLPs[target_name]
                else:
                    out = self.MLPs[target_name](out_embed)
                y = get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    # print('target', target.shape)
                    # if FLAGS.loss == 'RMSE':
                    #     loss = torch.sqrt(self.loss_function(out, target))
                    #     # loss = mean_squared_error(target, out, squared=False)
                    # elif FLAGS.loss == 'MSE' or FLAGS.loss == 'myGNLL':
                    loss = self.loss_function(
                        out, target
                    )  ## predicting my and log_var in case of myGNLL
                    # loss = mean_squared_error(target, out, squared=True)
                    # elif FLAGS.loss == 'GNLL':  ## predicting var
                    #     var = self.my_softplus(out[:, 1].reshape(-1, 1))
                    #     loss = self.loss_function(out[:, 0].reshape(-1, 1), target, var)
                    # else:
                    #     raise NotImplementedError()
                    # print('loss', loss.shape)
                else:
                    target = y.view((len(y)))
                    loss = self.loss_function(out, target)

                    
                out_dict[target_name] = out

                if FLAGS.pairwise_class and FLAGS.loss_components_to_keep != 'all' and target_name not in FLAGS.loss_components_to_keep:
                    saver.log_info_once(f'Ignore loss component {target_name} (but still in out_dict for test() to work)')
                    continue

                total_loss += loss
                loss_dict[target_name] = loss

        if (
            FLAGS.pairwise_class
            and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            and FLAGS.loss_components in ['both', 'class_only']
            and (
                forward_pairwise
                or (
                    not forward_pairwise
                    and FLAGS.subtask == 'train'
                    and 'per_batch' in FLAGS.pairwise_train_schemes
                    and 'node_att_diff' not in FLAGS.comp_ops # tricky code: turn off pairwise comp if this model is being trained with the regular data loader yet this op requires same-kernel comp
                )
            )
        ):
            pairwise_comp_results = []

            if forward_pairwise:
                strict_even = True
            else:
                strict_even = False

            d1_gemb, d2_gemb = split_vec_mat_into_2_halves(
                out_embed, strict_even=strict_even
            )
            for op in FLAGS.comp_ops:
                if op == 'hadamard':
                    c = d1_gemb * d2_gemb
                elif op == 'diff':
                    c = d1_gemb - d2_gemb
                elif op == 'emb_d1':
                    c = d1_gemb
                elif op == 'emb_d2':
                    c = d2_gemb
                elif op == 'node_att_diff':
                    c = self.node_att_diff(out_gnn, batch)
                else:
                    raise NotImplementedError()
                pairwise_comp_results.append(c)
            pairwise_comp_results = torch.cat(pairwise_comp_results, dim=1)

            total_loss += self._apply_target_MLPs_with_loss(
                self.MLPs_pairwise_class,
                pairwise_comp_results,
                data,
                out_dict,
                loss_dict,
                'pairwise_class',
                strict_even
            )

        #     print(
        #         f'pairwise loss added! total_loss={total_loss}; loss_dict={loss_dict}'
        #     )

        # else:
        #     print(f'pairwise loss NOT added!')


        return out_dict, total_loss, loss_dict, gae_loss

    def _apply_target_MLPs_with_loss(
        self, mlps, out_embed, data, out_dict, loss_dict, mode, strict_even
    ):
        loss_rtn = 0.0
        if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
            out_MLPs = mlps(out_embed)
        for target_name in self.target_list:
            if mode == 'normal':
                target_name_s = target_name
            elif mode == 'pairwise_class':
                target_name_s = f'{target_name}_pairwise_class'
            else:
                assert False

            if FLAGS.loss_components_to_keep != 'all' and target_name_s not in FLAGS.loss_components_to_keep:
                saver.log_info_once(f'Ignore loss component {target_name_s}')
                continue

            if hasattr(self, "MLP_version") and self.MLP_version == 'multi_obj':
                out = out_MLPs[target_name]
            else:
                out = mlps[target_name](out_embed)

            y = get_y_with_target(data, target_name)
            if self.task == 'regression':
                target = y.view((len(y), self.out_dim))
            else:
                target = y.view((len(y)))

            if mode == 'normal':
                # if self.task == 'regression' and FLAGS.loss_func == 'mse_weighted_util':
                #     loss = self.loss_function(out, target, target_name=target_name)
                # else:
                loss = self.loss_function(out, target)
                if (
                    hasattr(data, 'weight_switch')
                    and data.weight_switch == True
                    and 'util' in target_name
                ):
                    # weight = torch.clamp(1 - torch.sigmoid(-y+0.8), min=0.1)
                    weight = torch.clamp(
                        1 - torch.sigmoid(-out + 0.8), min=0.001
                    ).detach()
                    loss = weight * loss
                loss = loss.mean()
            elif mode == 'pairwise_class':
                # y1, y2 = split_vec_mat_into_2_halves(target, strict_even=strict_even)
                # target_pairwise = get_comp_result_tagret(y1, y2)
                # assert out.shape[0] == target_pairwise.shape[0]
                # loss = self.loss_function_pairwise_class(out, target_pairwise)
                loss = self.loss_function_pairwise_class(out, target, loss_dict, strict_even=strict_even)
            else:
                assert False

            if FLAGS.loss_scale is not None:
                loss = loss * FLAGS.loss_scale[target_name]

            if FLAGS.margin_loss:
                sorted_out = out[torch.argsort(target, dim=0)].view(out.shape)
                shifted_delta = (sorted_out - torch.roll(sorted_out, -1, 0))[0:-1]
                margin_loss = torch.mean(
                    torch.max(
                        torch.zeros(shifted_delta.shape),  # .to(FLAGS.device),
                        shifted_delta,
                    )
                )
                print('margin loss', margin_loss)
                loss_rtn += margin_loss
            out_dict[target_name_s] = out

            # use_regression_loss = True
            # if (
            #     FLAGS.pairwise_class
            #     and 'pariwise_comparison' in FLAGS.pairwise_what_branches
            #     and FLAGS.loss_components in ['both', 'class_only']
            # ):
            #     use_regression_loss = False

            # if use_regression_loss:
            loss_rtn += loss
            loss_dict[target_name_s] = loss
            # print(target_name, 'loss', loss)


            if mode == 'pairwise_class':
                if FLAGS.branch_discrepancy_loss:
                    y1_pred, y2_pred = split_vec_mat_into_2_halves(out_dict[target_name], strict_even=strict_even)
                    f_point = torch.cat([y1_pred, y2_pred], dim=1)
                    f_comp = out_dict[target_name_s]
                    saver.log_info_at_most(f'f_point={f_point}', f'bcl_fp', 1)
                    saver.log_info_at_most(f'f_comp={f_comp}', f'bcl_fc', 1)
                    loss = self._branch_discrepancy_loss(f_point, f_comp)
                    loss_dict[f'{target_name_s}_branch_discrepancy'] = loss
                    loss_rtn += loss
                    saver.log_info_at_most(f'loss_dict={loss_dict}', f'bcl_ld', 1)

            if FLAGS.listmle_loss: # applied to the pointwise branch
                predicted = out_dict[target_name]
                true = target
                saver.log_info_at_most(f'predicted={predicted}', f'lmle_p', 5)
                saver.log_info_at_most(f'true={true}', f'lmle_t', 5)
                loss = self._listMLE_loss(predicted, true)
                loss_dict[f'{target_name_s}_listmle'] = loss
                loss_rtn += loss
                saver.log_info_at_most(f'ListMLE loss={loss}', f'lmle_p', 5)

                    # exit()
        return loss_rtn
    
    def _branch_discrepancy_loss(self, f_point, f_comp):
        # f_point: [batch_size, 2] (pointwise predictions for two designs in each pair)
        # f_comp: [batch_size, 2] (logits for pairwise comparisons)
        
        # Calculate the absolute difference of pointwise predictions
        delta = torch.abs(f_point[:, 0] - f_point[:, 1])  # [batch_size]

        # Calculate softmax probabilities from pairwise logits
        softmax_prob = F.softmax(f_comp, dim=1)  # [batch_size, 2]

        # Determine the wrong class based on pointwise predictions
        wrong_class = (f_point[:, 0] > f_point[:, 1]).long()  # [batch_size]
        
        # Discrepancy loss
        discrepancy_loss = softmax_prob[torch.arange(softmax_prob.size(0)), wrong_class] * delta

        # Hyperparameters to balance the loss components
        alpha = 1.0

        # Combined loss
        total_loss = alpha * torch.mean(discrepancy_loss)# + beta * torch.mean(consistency_reward)

        return total_loss
    
    def _listMLE_loss(self, predicted, true):
        """
        Computes ListMLE loss for given predictions and true values.
        Args:
        predicted (torch.Tensor): Predicted values from the model.
        true (torch.Tensor): True values corresponding to the predictions.

        Returns:
        float: Computed ListMLE loss.
        """
        _, sorted_indices = torch.sort(true, descending=True)
        predicted_sorted = predicted.gather(0, sorted_indices)
        # return torch.sum(torch.logcumsumexp(-predicted_sorted, dim=0)) - torch.sum(-predicted_sorted)
        log_cumsum = torch.logcumsumexp(-predicted_sorted, dim=0)
        loss = torch.sum(log_cumsum) - torch.sum(-predicted_sorted)
        normalized_loss = loss / predicted_sorted.size(0)  # Normalizing by the length of the list
        return normalized_loss


    def _create_target_MLPs_dict(self, target_list, hidden_channels, in_D, out_D):

        if (not hasattr(self, "MLP_version")) or self.MLP_version == 'single_obj':
            MLPs = nn.ModuleDict()
            for target in self.target_list:
                MLPs[target] = MLP(
                    in_D,
                    out_D,
                    activation_type=FLAGS.activation,
                    hidden_channels=hidden_channels,
                    num_hidden_lyr=len(hidden_channels),
                )
        else:
            MLPs = MLP_multi_objective(
                in_D,
                out_D,
                activation_type=FLAGS.activation,
                hidden_channels=hidden_channels,
                objectives=self.target_list,
                num_common_lyr=FLAGS.MLP_common_lyr,
            )

        """ #commeted by zongyue
        rtn = nn.ModuleDict()
        
        for target in target_list:
            rtn[target] = MLP(in_D, out_D, activation_type=FLAGS.activation,
                              hidden_channels=hidden_channels,
                              num_hidden_lyr=len(hidden_channels))
        return rtn
        """
        return MLPs


