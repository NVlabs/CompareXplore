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

from saver import saver
from data import get_num_features, get_edge_dim
from config import FLAGS
from model import Net, HARPNet
from model_multi_modality import MultiModalityNet
from model_code2vec import Code2VecNet
from model_openai import OpenAIModel
from model_pytriton import PytritonModel
from pretrained_GNN_handler import create_and_load_zongyue_pretrained_GNN


def create_model(dataset, data_loader, task=FLAGS.task, load_model=None, D=FLAGS.D):
    # saver.log_info(f'edge_dim={edge_dim}')
    # if FLAGS.dataset == 'simple-programl' or FLAGS.target_kernel is not None:
    #     init_pragma_dict = {'all': [1, 21]}
    pragma_dim = dataset.get_attribute('init_feat_dict')
    if FLAGS.model == 'code2vec':
        c = Code2VecNet
    elif FLAGS.model == 'OpenAI':
        c = OpenAIModel
    elif FLAGS.model == 'pytriton':
        c = PytritonModel
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            c = MultiModalityNet
        else:
            c = Net
    if load_model is None:
        load_model = FLAGS.load_model
    if 'atefeh' in load_model or FLAGS.load_model_HARPnet:
        c = HARPNet
        model = c(in_channels=153, edge_dim=get_edge_dim(dataset), task=task, init_pragma_dict=pragma_dim, num_layers=6, D=D)#.to(FLAGS.device)
    else:
        model = c(task=task, init_pragma_dict=pragma_dim, dataset=dataset, D=D)#.to(FLAGS.device)
    if FLAGS.load_pretrained_GNN:
        model.pretrained_GNN_encoder = create_and_load_zongyue_pretrained_GNN()#.to(FLAGS.device)

    # if data_loader is not None:
    #     # Dummy run.
    #     # Otherwise, may get this error "RuntimeError: Modules with uninitialized parameters can't be used with `DistributedDataParallel`. Run a dummy forward pass to correctly initialize the modules
    #     # (probably due to num_features == 0 so that somehow there is lazy initialization by torch for conv_first)
    #     data = next(iter(data_loader))
    #     model(data, forward_pairwise=False, tvt='test')
        
    saver.log_info_once(f'Model factory:\n{model}')
    
    if FLAGS.mode == "acc_launch":
        pass
    else:
        assert FLAGS.mode == "standalone" 
        model = model.to(FLAGS.device) # do it here instead outside to avoid subtle issues

    return model
