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

from nn import MyGlobalAttention
from config import FLAGS
import torch
import torch.nn as nn
from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU
from utils import MLP

def create_graph_att_module(D, return_gate_nn=False):
    def _node_att_gate_nn(D):
        if FLAGS.node_attention_MLP:
            return MLP(D, 1,
                       activation_type=FLAGS.activation_type,
                       hidden_channels=[D // 2, D // 4, D // 8],
                       num_hidden_lyr=3)
        else:
            return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    gate_nn = _node_att_gate_nn(D)
    glob = MyGlobalAttention(gate_nn, None)
    if return_gate_nn:
        return gate_nn, glob
    else:
        return glob


class LpModule(torch.nn.Module):
    def __init__(self, p):
        super(LpModule, self).__init__()
        self.p = p

    def forward(self, x, **kwargs):
        x = x / (torch.norm(x, p=self.p, dim=-1).unsqueeze(-1) + 1e-12)
        return x
