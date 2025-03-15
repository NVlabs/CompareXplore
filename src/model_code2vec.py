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

from model import Model
from config import FLAGS
import torch

from collections import OrderedDict


class Code2VecNet(Model):
    def __init__(self, init_pragma_dict=None, dataset=None, task=None):
        super(Code2VecNet, self).__init__()
        self.task = FLAGS.task

        self.out_dim, self.loss_function = self._create_loss()

        self.target_list = self._get_target_list()

        self.decoder, _ = self._create_decoder_MLPs(384, 384, self.target_list,
                                                    self.out_dim, hidden_channels=None)

    def forward(self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None):
        total_loss = 0.0
        out_dict = OrderedDict()
        loss_dict = OrderedDict()

        total_loss += self._apply_target_MLPs_with_loss(self.decoder, data.x.float(), data, out_dict,
                                          loss_dict,
                                          'normal')

        return out_dict, total_loss, loss_dict, torch.tensor(0.0)
