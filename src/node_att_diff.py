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

from config import FLAGS
from saver import saver
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add


class NodeAttDiff(nn.Module):
    def __init__(self, node_dim):
        super(NodeAttDiff, self).__init__()
        if FLAGS.node_att_diff_cat_both:
            factor = 3
        else:
            factor = 2
        self.attention_net = nn.Sequential(
            nn.Linear(factor * node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, 1)
        )

    def forward(self, out_gnn, batch_input):
        batch = batch_input[0 : len(batch_input) // 2]

        # saver.log_info(f'batch_input={batch_input}')
        # saver.log_info(f'batch_input.shape={batch_input.shape}')

        node_embed_1, node_embed_2 = split_vec_mat_into_2_halves(
            out_gnn, strict_even=True
        )

        # Compute attention scores and differences
        attention_scores, diff_embeddings = self._compute_att(
            node_embed_1, node_embed_2, batch
        )

        if FLAGS.node_att_diff_scaled_att:
            # Apply differentiable Top-K using softmax with scaling (a soft version of Top-K)
            scaled_attention_scores = F.softmax(  # TODO: is this correct or needed?
                attention_scores * 10, dim=0
            )  # Scaling factor of 10 to sharpen the focus
        else:
            scaled_attention_scores = attention_scores

        # saver.log_info(f'attention_scores shape: {attention_scores.shape}')
        # saver.log_info(f'scaled_attention_scores shape: {scaled_attention_scores.shape}')

        # saver.log_info(f'attention_scores: {attention_scores}')
        # saver.log_info(f'scaled_attention_scores: {scaled_attention_scores}')

        # exit()
        # Weight differences by scaled attention scores
        weighted_diff_embeddings = (
            scaled_attention_scores.unsqueeze(-1) * diff_embeddings
        )

        # Sum up the weighted differences per graph
        out_embed = scatter_add(weighted_diff_embeddings, batch, dim=0)

        return out_embed

    def _compute_att(self, node_embed_1, node_embed_2, batch):

        # Concatenate node embeddings with their differences
        diff_embeddings = node_embed_1 - node_embed_2

        if FLAGS.node_att_diff_cat_both:
            concat_embeddings = torch.cat([node_embed_1, node_embed_2, diff_embeddings], dim=-1)
        else:
            concat_embeddings = torch.cat([node_embed_1, diff_embeddings], dim=-1)

        # Compute raw attention scores
        raw_scores = self.attention_net(concat_embeddings).squeeze(-1)

        # Apply softmax per graph in the batch

        # saver.log_info(f'raw_scores.shape={raw_scores.shape}')
        # saver.log_info(f'batch.shape={batch.shape}')
        # saver.log_info(f'batch={batch}; batch.dtype={batch.dtype}')
        # saver.log_info(f'raw_scores={raw_scores}; raw_scores.dtype={raw_scores.dtype}')

        # assert (
        #     batch.min() >= 0 and batch.max() < batch.max() + 1
        # ), "Invalid batch indices"
        # assert (
        #     raw_scores.shape == batch.shape
        # ), "Shape mismatch between raw_scores and batch"

        # saver.log_info(
        #     f'Min batch index: {batch.min()}, Max batch index: {batch.max()}'
        # )
        # saver.log_info(
        #     f'node_embed_1 shape: {node_embed_1.shape}, node_embed_2 shape: {node_embed_2.shape}'
        # )

        # saver.log_info(torch.isnan(raw_scores).any())
        # saver.log_info(torch.isinf(raw_scores).any())
        # num_graphs = batch.max() + 1
        # assert batch.min() >= 0 and batch.max() < num_graphs

        # max_scores = scatter_max(raw_scores.cpu(), batch.cpu(), dim=0)
        # saver.log_info(f'1 max_scores cpu: {max_scores}')
        # max_scores = max_scores[0]
        # saver.log_info(f'2 max_scores cpu: {max_scores}')
        # max_scores = max_scores[batch.cpu()]
        # saver.log_info(f'3 max_scores cpu: {max_scores}')

        # According to GPT-4,
        # Device Synchronization: Before the call to scatter_max, ensure that all prior GPU operations are completed. Sometimes, asynchronous execution can lead to operations being queued up in ways that cause conflicts. You can force operations to complete with torch.cuda.synchronize() before the call to scatter_max to see if this resolves the issue.
        if hasattr(FLAGS, 'cuda_synchronize_trick') and FLAGS.cuda_synchronize_trick:
            torch.cuda.synchronize()
        max_scores = scatter_max(raw_scores, batch, dim=0) # weird line that may throw RuntimeError: CUDA error: an illegal memory access was encountered
        # saver.log_info(f'1 max_scores: {max_scores}')
        max_scores = max_scores[0]
        # saver.log_info(f'2 max_scores: {max_scores}')
        max_scores = max_scores[batch]
        # saver.log_info(f'3 max_scores: {max_scores}')

        exp_scores = torch.exp(raw_scores - max_scores)

        # saver.log_info(f'exp_scoress: {exp_scores}')

        exp_sum = scatter_add(exp_scores, batch, dim=0)[batch]

        # saver.log_info(f'exp_sum: {exp_sum}')

        attention_scores = exp_scores / exp_sum

        return attention_scores, diff_embeddings


def split_vec_mat_into_2_halves(input, strict_even=True):
    length = input.shape[0]
    if strict_even:
        assert (
            length % 2 == 0
        ), f'length={length}'  # divisible by 2 -- otherwise data loader has some issue
    half_point = int(length / 2)
    d1 = input[0:half_point]
    if strict_even:
        d2 = input[half_point:]
    else:
        d2 = input[half_point : 2 * half_point]
    assert d1.shape == d2.shape
    return d1, d2
