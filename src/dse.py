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


from config import FLAGS
from saver import saver
from utils import (
    get_root_path,
    _get_y_with_target,
    OurTimer,
    create_dir_if_not_exists,
    HARP_model_loaded,
    deduce_load_model_path,
    print_list_of_numbers_with_perc,
    print_stats,
    get_file_size_str,
)
from data import (
    SAVE_DIR,
    read_gexf_file,
    encode_node_edge_dict,
    encode_X_torch,
    GEXF_FILES,
    create_dataloader,
    torch_generator,
)

from model_factory import create_model
from parameter import (
    DesignSpace,
    DesignPoint,
    get_default_point,
    topo_sort_param_ids,
    compile_design_space,
    gen_key_from_design_point,
)
from config_ds import build_config
from result import Result
from pairwise import convert_out_logits_to_rank_scores, get_pairwise_data_loader

import json
import os
from math import ceil, inf
from os.path import join, basename

import time
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Union, Generator, Any
import sys
import networkx as nx
from collections import OrderedDict
from glob import glob, iglob
import pickle
import shutil
import numpy as np

# from harp_dse import HARPGNNModel
# from database import create_database
import torch.nn.functional as F
import heapq
import random
from collections import defaultdict
from tqdm import tqdm


class ModelWrapper:
    def __init__(
        self,
        dataset,
        path,
        model_info_collector,
        first_dse=False,
        task='regression',
        adapt_result_dict=None,
        kernel_name=None,
        load_model=None,
        model_type=None,
    ):

        self.log = saver
        self.path = path
        self.model_info_collector = model_info_collector
        self.model_type = model_type

        if adapt_result_dict is not None:
            adapted_models = adapt_result_dict.get('adapted_models_dict').get(
                kernel_name
            )
            if adapted_models is None:
                raise RuntimeError(
                    f'No adapted models in a non-None adapt_result_dict for kernel {kernel_name}: {adapt_result_dict}'
                )

            # chosen_model = adapted_models[0]  # TODO

            self.models = adapted_models
            # saver.info(f'Chosen model from {len(adapted_models)} adapted models in adapt_result_dict')
        else:
            if task == 'regression':
                self.model_path = FLAGS.load_model
                D  = FLAGS.D
            else:
                self.model_path = FLAGS.load_model_class
                D = 64 # TODO: hakcy; always use D=64 for now but can change later

            self.model = create_model(
                dataset, None, task=task, load_model=self.model_path, D=D
            )

            if first_dse:
                saver.log_model_architecture(self.model)

            if load_model is None:
                load_model = deduce_load_model_path(task, FLAGS)
                # model = saver.accelerator.prepare(
                #     model
                # )
            else:
                self.log.info(f'load_model is specified; load_model={load_model}')

            if load_model != None and load_model != 'None':

                self.model, loaded_model_info = saver.load_trained_model(
                    load_model, self.model
                )

                saver.info(
                    f'Loaded {task} model from {load_model}; loaded_model_info: {loaded_model_info}'
                )

            self.models = [self.model]
        # self.encoder = load(self.encoder_path)
        self.encoder, self.preprocessors = dataset.encoders, dataset.preprocessors

    def perf_as_quality(self, new_result: Result) -> float:
        """Compute the quality of the point by (1 / latency).

        Args:
            new_result: The new result to be qualified.

        Returns:
            The quality value. Larger the better.
        """
        return 1.0 / new_result.perf

    def test(
        self, loader, data_list, config, mode='regression', forward_pairwise=False
    ):
        results_across_models = []
        assert len(self.models) >= 1
        for model in self.models:
            model.eval()

            i = 0
            results: List[Result] = []
            target_list = FLAGS.target
            pairwise_probs_list = []
            if not isinstance(FLAGS.target, list):
                target_list = [FLAGS.target]

            if forward_pairwise:
                loader_use = tqdm(loader)
            else:
                loader_use = loader
            for data in loader_use:
                if FLAGS.mode == 'acc_launch':
                    pass
                else:
                    assert FLAGS.mode == 'standalone'
                    data = data.to(FLAGS.device)

                with torch.no_grad():

                    timer = OurTimer()
                    out_dict, loss, loss_dict_, _ = model(
                        data,
                        forward_pairwise=forward_pairwise,
                        tvt='test',
                        iter=-1,
                        test_name='',
                    )

                    # saver.log_info(f'out_dict={out_dict}')

                    self.model_info_collector.collect_running_time(
                        timer.time_and_clear(only_seconds=True), self.model_type
                    )

                if mode == 'regression':

                    if forward_pairwise:
                        # probs = F.softmax(out_dict['perf_pairwise_class'], dim=1)
                        probs = convert_out_logits_to_rank_scores(
                            out_dict['perf_pairwise_class']
                        )
                        pairwise_probs_list.append(probs)
                    else:
                        for i in range(len(out_dict['perf'])):
                            curr_result = Result()
                            curr_result.point = data_list[i].point

                            # curr_result.data = deepcopy(data_list[i])
                            curr_result.data = data_list[i]

                            for target_name in target_list:
                                out = out_dict[target_name]
                                out_value = out[i].item()
                                if target_name == 'perf':
                                    curr_result.perf = out_value
                                    if FLAGS.encode_log:
                                        curr_result.actual_perf = 2**out_value
                                    else:
                                        curr_result.actual_perf = out_value
                                elif target_name in curr_result.res_util.keys():
                                    curr_result.res_util[target_name] = out_value
                                else:
                                    raise NotImplementedError()
                            curr_result.quality = self.perf_as_quality(curr_result)

                            # prune if over-utilizes the board
                            max_utils = config['max-util']
                            utils = {
                                k[5:]: max(0.0, u)
                                for k, u in curr_result.res_util.items()
                                if k.startswith('util-')
                            }
                            # utils['util-LUT'] = 0.0
                            # utils['util-FF'] = 0.0
                            # utils['util-BRAM'] = 0.0
                            if FLAGS.prune_util:
                                curr_result.valid = all(
                                    [
                                        (utils[res] / FLAGS.util_normalizer)
                                        < max_utils[res]
                                        for res in max_utils
                                    ]
                                )
                                # curr_result.valid = all([(utils[res] / FLAGS.util_normalizer )< 0.74 for res in max_utils])
                            else:
                                curr_result.valid = True
                            results.append(curr_result)
                elif mode == 'class':
                    _, pred = torch.max(out_dict['perf'], 1)
                    labels = _get_y_with_target(data, 'perf')
                    valid_or_not = pred == labels
                    results = valid_or_not.tolist()
                    if len(results) != 1:
                        raise RuntimeError(
                            f'Assume the classifcation mode only takes 1 design at a time!'
                        )
                else:
                    assert False

            results_across_models.append(results)

            if mode == 'regression' and forward_pairwise:
                pairwise_probs = torch.cat(pairwise_probs_list, dim=0)
                assert len(self.models) == 1, f'TODO: handle multiple models'
                return pairwise_probs

        if mode == 'regression':
            rtn = []
            num_results = None
            for results in results_across_models:
                if num_results is None:
                    num_results = len(results)
                else:
                    assert num_results == len(results)
            for i in range(num_results):
                results_to_aggregate = []
                for results in results_across_models:
                    results_to_aggregate.append(results[i])
                rtn.append(Result.aggregate_results(results_to_aggregate))

            assert len(rtn) == len(results_across_models[0])

        elif mode == 'class':
            results_to_aggr = []
            for r in results_across_models:
                assert len(r) == 1
                results_to_aggr.append(r[0])
            rtn = Result.majority_voting(results_to_aggr)
            assert rtn == True or rtn == False
            return rtn
        else:
            assert False

        return rtn


class Explorer:
    def __init__(
        self,
        dataset,
        dataset_class,
        kernel_name: str,
        first_dse: bool = False,
        adapt_result_dict=None,
        adapt_result_dict_class=None,
    ):
        """Constructor.

        Args:
            ds: DesignSpace
        """
        self.log = saver
        assert type(kernel_name) is str
        self.kernel_name = kernel_name

        config_path = None
        for dataset_str in ['machsuite', 'poly']:
            path_kernel = join(get_root_path(), 'dse_database', dataset_str, 'config')
            cp = join(path_kernel, f'{kernel_name}_ds_config.json')
            print(cp)
            if os.path.exists(cp):
                if config_path is None:
                    config_path = cp
                else:
                    raise RuntimeError(
                        f'Duplicate config paths for kernel {kernel_name}: {config_path} and {cp}'
                    )

        self.config_path = config_path
        self.config = self.load_config()

        # self.timeout = self.config['timeout']['exploration']
        # self.timeout = float(inf)
        self.timeout = FLAGS.dse_timeout
        # self.hls_timeout = 40
        self.ds, self.ds_size = compile_design_space(
            self.config['design-space']['definition'], None, self.log
        )

        self.batch_size = 1
        # Status checking
        self.num_top_designs = FLAGS.num_top_designs

        if FLAGS.pairwise_comp_DSE:
            assert (
                FLAGS.num_cand_for_ver >= FLAGS.num_top_designs
            ), f'First prune down to num_cand_for_ver then verify these designs and get down to num_top_designs'

            assert FLAGS.num_designs_to_cache_during_search >= FLAGS.num_cand_for_ver

            self.num_top_designs = FLAGS.num_designs_to_cache_during_search

        assert self.num_top_designs > 0
        self.key_perf_dict = OrderedDict()
        self.best_results_dict = {}
        self.best_result: Result = Result()
        self.explored_point = 0
        self.ordered_pids = self.topo_sort_param_ids(self.ds)
        # self.ensemble_GNNmodels = []  ## list of GNN models for regression. if ensemble is not used, only has one entry
        # self.ordered_pids = FLAGS.ordered_pids
        self.model_info_collector = ModelInfoCollector()

        self.model_wrapper = ModelWrapper(
            dataset,
            SAVE_DIR,
            self.model_info_collector,
            first_dse=first_dse,
            task='regression',
            adapt_result_dict=adapt_result_dict,
            kernel_name=kernel_name,
            model_type='regression_model',
        )
        self.dataset = dataset
        self.dataset_class = dataset_class

        found_file = None
        for gexf_file in GEXF_FILES:
            bn = basename(gexf_file)
            if '_processed_result' in bn:  # , f'bn={bn}\ngexf_file={gexf_file}'
                kn = bn.split('_processed_result')[0]
            else:
                assert '.gexf' in bn
                kn = bn.split('.gexf')[0]
            if kn == 'stencil_stencil2d':
                kn = 'stencil'
            if kernel_name == kn:
                if found_file is None:
                    found_file = gexf_file
                else:
                    raise RuntimeError(
                        f'Already found file {found_file} but another file {gexf_file} -- check {kernel_name}'
                    )

        print(found_file)
        assert (
            found_file is not None
        ), f'Cannot find kernel_name {kernel_name} in {len(GEXF_FILES)} GEXF_FILES: {GEXF_FILES}'
        self.graph_path = found_file
        saver.info(f'graph path {self.graph_path}')
        self.graph = read_gexf_file(
            self.graph_path, sequence_modeling=FLAGS.sequence_modeling
        )
        if HARP_model_loaded(FLAGS):
            self.graph_class = read_gexf_file(
                self.graph_path, multi_modality=False, sequence_modeling=False
            )
        else:
            self.graph_class = self.graph

        ## for ploting one of the objectives (all points)
        self.plot_data = {k: [] for k in FLAGS.target}

        self.prune_invalid = FLAGS.prune_invalid
        if self.prune_invalid:
            self.model_wrapper_valid = ModelWrapper(
                dataset_class,
                SAVE_DIR,
                self.model_info_collector,
                task='class',
                adapt_result_dict=adapt_result_dict_class,
                kernel_name=kernel_name,
                model_type='classification_model',
            )
            self.cnt_invalid_pruned = 0
            self.cnt_valid = 0

        if FLAGS.pairwise_comp_DSE:
            self.model_wrapper_pairwise = ModelWrapper(
                dataset,
                SAVE_DIR,
                self.model_info_collector,
                first_dse=first_dse,
                task='regression',
                adapt_result_dict=adapt_result_dict,
                kernel_name=kernel_name,
                load_model=FLAGS.load_model_pairwise,
                model_type='pairwise_model',
            )

        self.cached_edge_index = None
        self.cached_edge_attr = None

    def load_config(self) -> Dict[str, Any]:
        """Load the DSE configurations.

        Returns:
            A dictionary of configurations.
        """

        try:
            if not os.path.exists(self.config_path):
                self.log.error(('Config JSON file not found: %s', self.config_path))
                raise RuntimeError()

            self.log.info('Loading configurations')
            with open(self.config_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error(('Failed to load config: %s', str(err)))
                    raise RuntimeError()

            config = build_config(user_config, self.log)
            if config is None:
                self.log.error(('Config %s is invalid', self.config_path))
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)

        return config

    def get_pragmas(self, point: DesignPoint) -> List[int]:
        pragmas = []
        for _, value in sorted(point.items()):
            if type(value) is str:
                if value.lower() == 'flatten':
                    value = 100  # 2
                elif value.lower() == 'off':
                    value = 1
                elif value.lower() == '':
                    value = 50  # 3
                else:
                    print(value)
                    raise ValueError()
            elif type(value) is int:
                pass
            else:
                raise ValueError()
            pragmas.append(value)
        return pragmas

    def apply_design_point(self, g, point: DesignPoint, mode='regression') -> Data:
        sequence_modeling = FLAGS.sequence_modeling
        multi_modality = FLAGS.multi_modality
        if mode == 'regression':
            dataset = self.dataset
        else:
            assert mode == 'class'
            dataset = self.dataset_class
            if HARP_model_loaded(
                FLAGS
            ):  # very tricky code: need to do this special thing for HARP!
                sequence_modeling = False
                multi_modality = False

        d_node, edge_dict = encode_node_edge_dict(
            g,
            dataset.preprocessors,
            point,
            multi_modality=multi_modality,
            sequence_modeling=sequence_modeling,
        )

        # X, d_node = model.encode_node(g, point)
        # edge_attr = model.encode_edge(g)
        # edge_index = create_edge_index(g)
        pragmas = self.get_pragmas(point)

        # d_node = dict()
        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        keys = ['perf', 'actual_perf', 'quality']
        d_node['pragmas'] = torch.FloatTensor(np.array([pragmas]))

        d_node['kernel_speedup'] = torch.FloatTensor(np.array([-1]))  # unknown

        # d_node['X_contextnids'] = X_contextnids
        # d_node['X_pragmanids'] = X_pragmanids
        # d_node['X_pseudonids'] = X_pseudonids
        # d_node['X_icmpnids'] = X_icmpnids
        for r in resources:
            keys.append('util-' + r)
            keys.append('total-' + r)
        for key in keys:
            d_node[key] = 0
        if mode == 'class':  ## default: point is valid
            d_node['perf'] = 1

        d_node['point'] = point

        data = encode_X_torch(
            g,
            d_node,
            edge_dict,
            dataset.preprocessors,
            '',
            '',
            multi_modality=multi_modality,
            sequence_modeling=sequence_modeling,
            cached_edge_index=self.cached_edge_index,
            cached_edge_attr=self.cached_edge_attr,
        )

        if self.cached_edge_index is None:
            self.cached_edge_index = data.edge_index
            assert self.cached_edge_attr is None
            self.cached_edge_attr = data.edge_attr
            self.log.info(f'Updated self.cached_edge_index and self.cached_edge_attr')

        data.point = point
        return data

    def update_best(self, result: Result) -> None:
        """Keep tracking the best result found in this explorer.

        Args:
            result: The new result to be checked.

        """
        # if result.valid and result.quality > self.best_result.quality:
        if 'speedup' in FLAGS.norm_method:
            REF = min
        else:
            REF = max
        if self.key_perf_dict and len(self.key_perf_dict) >= self.num_top_designs:
            key_refs_perf = REF(
                self.key_perf_dict, key=(lambda key: self.key_perf_dict[key])
            )
            refs_perf = self.key_perf_dict[key_refs_perf]
        else:
            if REF == min:
                refs_perf = float(-inf)
            else:
                refs_perf = float(inf)
        point_key = gen_key_from_design_point(result.point)
        updated = False
        if (
            point_key not in self.key_perf_dict
            and result.valid
            and REF(result.perf, refs_perf) != result.perf
        ):  # if the new result is better than the references designs
            ## use the below condition when all the perf numbers are the same, such as for aes
            # if result.valid and (REF(result.perf, refs_perf) != result.perf or refs_perf == result.perf): # if the new result is better than the references designs
            # if result.valid and (not self.key_perf_dict or self.key_perf_dict[max(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))] < result.perf): # if the new result is better than the references designs
            self.best_result = result
            self.log.debug(
                (
                    'Found a better result at {}: Quality {:.1e}, Perf {:.1e}'.format(
                        self.explored_point, result.quality, result.perf
                    )
                )
            )
            if len(self.key_perf_dict.keys()) >= self.num_top_designs:
                ## replace maxmimum performance value
                key_refs_perf = REF(
                    self.key_perf_dict, key=(lambda key: self.key_perf_dict[key])
                )
                self.best_results_dict.pop(
                    (self.key_perf_dict[key_refs_perf], key_refs_perf)
                )
                self.key_perf_dict.pop(key_refs_perf)

            attrs = vars(result)
            self.log.debug(
                ', '.join(
                    "%s: %s" % item for item in attrs.items() if item[0] != 'data'
                )
            )

            self.key_perf_dict[point_key] = result.perf
            self.best_results_dict[(result.perf, point_key)] = result
            updated = True

        if self.key_perf_dict.values():
            reward = REF([-p for p in self.key_perf_dict.values()])
            return reward, updated
        else:
            return 0, updated

    def gen_options(
        self, point: DesignPoint, pid: str, default=False
    ) -> List[Union[int, str]]:
        """Evaluate available options of the target design parameter.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            A list of available options.
        """
        if default:
            dep_values = {dep: point[dep].default for dep in self.ds[pid].deps}
        else:
            dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        options = eval(self.ds[pid].option_expr, dep_values)
        if options is None:
            self.log.error(
                f'Failed to evaluate {self.ds[pid].option_expr} with dep {str(dep_values)}'
            )
            print('Error: failed to manipulate design points')
            sys.exit(1)

        return options

    def get_order(self, point: DesignPoint, pid: str) -> int:
        """Evaluate the order of the current value.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            The order.
        """

        if not self.ds[pid].order:
            return 0

        order = eval(
            self.ds[pid].order['expr'], {self.ds[pid].order['var']: point[pid]}
        )
        if order is None or not isinstance(order, int):
            self.log.warning(
                f'Failed to evaluate the order of {pid} with value {str(point[pid])}: {str(order)}'
            )
            return 0

        return order

    def update_child(self, point: DesignPoint, pid: str) -> None:
        """Check values of affected parameters and update them in place if it is invalid.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.
        """

        pendings = [
            child for child in self.ds[pid].child if self.validate_value(point, child)
        ]
        for child in pendings:
            self.update_child(point, child)

    def validate_point(self, point: DesignPoint) -> bool:
        """Check if the current point is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        changed = False
        for pid in point.keys():
            options = self.gen_options(point, pid)
            value = point[pid]
            if not options:  # All invalid (something not right), set to default
                self.log.warning(f'No valid options for {pid} with point {str(point)}')
                point[pid] = self.ds[pid].default
                changed = True
                continue

            if isinstance(value, int):
                # Note that we assume all options have the same type (int or str)
                cand = min(options, key=lambda x: abs(int(x) - int(value)))
                if cand != value:
                    point[pid] = cand
                    changed = True
                    continue

            if value not in options:
                point[pid] = self.ds[pid].default
                changed = True
                continue

        return changed

    def validate_value(self, point: DesignPoint, pid: str) -> bool:
        """Check if the current value is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        options = self.gen_options(point, pid)
        value = point[pid]
        if not options:  # All invalid (something not right), set to default
            self.log.warning(f'No valid options for {pid} with point {str(point)}')
            point[pid] = self.ds[pid].default
            return False

        if isinstance(value, int):
            # Note that we assume all options have the same type (int or str)
            cand = min(options, key=lambda x: abs(int(x) - int(value)))
            if cand != value:
                point[pid] = cand
                return True

        if value not in options:
            point[pid] = self.ds[pid].default
            return True
        return False

    def move_by(self, point: DesignPoint, pid: str, step: int = 1) -> int:
        """Move N steps of pid parameter's value in a design point in place.

        Args:
            point: The design point to be manipulated.
            pid: The target design parameter.
            step: The steps to move. Note that step can be positive or negatie,
                  but we will not move cirulatory even the step is too large.

        Returns:
            The actual move steps.
        """

        try:
            options = self.gen_options(point, pid)
            idx = options.index(point[pid])
        except (AttributeError, ValueError) as err:
            self.log.error(
                f'Fail to identify the index of value {point[pid]} of parameter {pid} at design point {str(point)}: {str(err)}'
            )
            print('Error: failed to manipulate design points')
            sys.exit(1)

        target = idx + step
        if target >= len(options):
            target = len(options) - 1
        elif target < 0:
            target = 0

        if target != idx:
            point[pid] = options[target]
            self.update_child(point, pid)
        return target - idx

    def get_results(self, next_points: List[DesignPoint]) -> List[Result]:
        data_list = []
        if self.prune_invalid:
            for point in next_points:
                data_list.append(
                    self.apply_design_point(self.graph_class, point, mode='class')
                )

            test_loader = create_dataloader(
                data_list,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=0,
                is_file_li=False,
                multi_modality=False,
            )
            # test_loader = DataLoader(data_list, batch_size=self.batch_size)  # TODO
            valid = self.model_wrapper_valid.test(
                test_loader, data_list, self.config['evaluate'], mode='class'
            )
            if valid == 0:
                # stop if the point is invalid
                self.log.debug(f'invalid point {point}')
                self.cnt_invalid_pruned += 1
                return [float(inf)]  # TODO: add batch processing
            else:
                self.cnt_valid += 1
                pass  # continue the operation below

        data_list = []
        for point in next_points:
            data_list.append(self.apply_design_point(self.graph, point))

        test_loader = create_dataloader(
            data_list,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
            is_file_li=False,
            multi_modality=FLAGS.multi_modality,
        )
        results = self.model_wrapper.test(
            test_loader, data_list, self.config['evaluate'], mode='regression'
        )
        return results

    def topo_sort_param_ids(self, space: DesignSpace) -> List[str]:
        return topo_sort_param_ids(space)

    def traverse(
        self, point: DesignPoint, idx: int
    ) -> Generator[DesignPoint, None, None]:
        """DFS traverse the design space and yield leaf points.

        Args:
            point: The current design point.
            idx: The current manipulated parameter index.

        Returns:
            A resursive generator for traversing.
        """

        if idx == len(self.ordered_pids):
            # Finish a point
            yield point
        else:
            yield from self.traverse(point, idx + 1)

            # Manipulate idx-th point
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)

    @staticmethod
    def clone_point(point: DesignPoint) -> DesignPoint:
        return dict(point)

    def run(self) -> None:
        """The main function of the explorer to launch the search algorithm.

        Args:
            algo_name: The corresponding algorithm name for running this exploration.
            algo_config: The configurable values for the algorithm.
        """
        raise NotImplementedError()


class ExhaustiveExplorer(Explorer):
    def __init__(
        self,
        dataset,
        dataset_class,
        kernel_name: str,
        first_dse: bool = False,
        point: DesignPoint = None,
        adapt_result_dict=None,
        adapt_result_dict_class=None,
    ):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ExhaustiveExplorer, self).__init__(
            dataset,
            dataset_class,
            kernel_name,
            first_dse,
            adapt_result_dict,
            adapt_result_dict_class,
        )
        self.batch_size = 1  # TODO: in future, may consider something fancier
        self.kernel_name = kernel_name

        if hasattr(FLAGS, 'input_pickle') and FLAGS.input_pickle is not None:
            with open(FLAGS.input_pickle, 'rb') as f:
                design_candidates = pickle.load(f)
                design_candidates = list(design_candidates.items())
                design_candidates.sort(key=lambda x: x[0][0], reverse=True)
                self.design_candidates = design_candidates
                saver.log_info(
                    f'ExhaustiveExplorer: Loaded {len(self.design_candidates)} design_candidates from {FLAGS.input_pickle}'
                )
        else:
            self.design_candidates = None

        self.log.info('Done init')

        self.run()
        # attrs = vars(self.best_result)
        self._save_best_results_dict(self.best_results_dict, 'final')

        print(saver.get_log_dir())

    def _save_best_results_dict(self, best_results_dict, name):
        self.log.info(f'{name} Best Results')
        # i = 1

        if hasattr(FLAGS, 'exp_name') and FLAGS.exp_name != '':
            folder = f'dse_{FLAGS.exp_name}_{name}'
        else:
            folder = f'dse_{name}'
        folder = join(saver.logdir, folder)
        create_dir_if_not_exists(folder)
        fn = join(folder, f'{self.kernel_name}.pkl')
        processed_data = [key[1] for key, value in sorted(best_results_dict.items(), key=lambda item: item[0][0], reverse=True)]
        with open(fn, 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            self.log.info(
                f'Saved best_results_dict of len {len(processed_data)} to {fn} {get_file_size_str(fn)}'
            )
        # for _, result in sorted(best_results_dict.items()):
        # data_temp = result.data
        # delattr(result, 'data')
        # attrs = vars(result)
        # self.log.info(f'Design {i}')
        # self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        # i += 1
        # result.data = data_temp

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        # pylint:disable=missing-docstring

        self.log.info('Launch exhaustive search algorithm')

        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.debug(f'Iteration {iter_cnt}')
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.debug(f'Next point: {str(next_points[-1])}')
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break

        self.log.info('No more points to be explored, stop.')

    def run(self) -> None:
        # pylint:disable=missing-docstring

        # Create a search algorithm generator
        gen_next = self.gen()
        our_timer = OurTimer()
        timer = time.time()
        cnt = 0

        self.log.info(f'while loop travesal of the design space starts!')
        should_continue, cond_time, cond_point = self._should_continue(timer)
        while should_continue:
            try:
                # Generate the next set of design points
                if self.design_candidates is None:
                    next_points = next(gen_next)
                else:
                    if cnt >= len(self.design_candidates):
                        break
                    # print(self.design_candidates[cnt])
                    # print(self.design_candidates[cnt][1].point)
                    point = self.design_candidates[cnt][1].point
                    for key in point.keys():
                        if isinstance(point[key], torch.Tensor):
                            point[key] = point[key].item()
                    # xxx = input()
                    next_points = [point]
                    cnt += 1
                # next_points = next(gen_next)
                self.log.debug(
                    f'The algorithm generates {len(next_points)} design points'
                )
            except StopIteration:
                break

            results = self.get_results(next_points)

            for r in results:
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(
                        ', '.join(
                            "%s: %s" % item
                            for item in attrs.items()
                            if item[0] != 'data'
                        )
                    )
                    _, updated = self.update_best(r)
            self.explored_point += len(results)
            should_continue, cond_time, cond_point = self._should_continue(timer)

        self.log.info(
            f'Explored {self.explored_point} points; time: in {our_timer.time_and_clear()}; cond_time={cond_time}, cond_point={cond_point}'
        )

        if self.prune_invalid:
            print_list_of_numbers_with_perc(
                [self.cnt_invalid_pruned, self.cnt_valid],
                '[#invalid designs, #valid designs]',
                self.log.info,
            )

        p = saver.save_dict_as_pickle(
            self.best_results_dict,
            f'best_results_dict_end_stage_1_{len(self.best_results_dict)}',
        )
        self.log.info(
            f'Saved self.best_results_dict of len {len(self.best_results_dict)} in {our_timer.time_and_clear()} {get_file_size_str(p)}'
        )
        best_results_dict_stage_1 = get_top_k_results(
            self.best_results_dict, FLAGS.num_top_designs
        )
        self._save_best_results_dict(best_results_dict_stage_1, 'stage_1')
        self.log.info(
            f'Saved best_results_dict_stage_1 of len {len(best_results_dict_stage_1)} in {our_timer.time_and_clear()}'
        )

        if FLAGS.pairwise_comp_DSE:
            self.log.info(f'Stage 2: Verification using pairwise comparison starts')
            # self.log.info(f'')

            results_dict = get_top_k_results(
                self.best_results_dict, FLAGS.num_cand_for_ver
            )

            len_before = len(results_dict)
            assert len_before <= FLAGS.num_cand_for_ver

            if len_before <= FLAGS.num_top_designs:
                self.log.info('No need to run any pairwise comparison')
            else:
                results_dict = self._pairwise_comp(results_dict)

            self.best_results_dict = results_dict
            self.log.info(
                f'After stage 2, len(results_dict): {len_before} --> {len(self.best_results_dict)}; time: in {our_timer.time_and_clear()}'
            )

        self.model_info_collector.print_stats()

    def _pairwise_comp(self, results_dict):

        designs_sorted = sorted(results_dict.keys())

        saver.log_info(f'Creating pair_list')
        pair_list = []
        for design_1 in tqdm(designs_sorted):
            for design_2 in designs_sorted:

                result_1 = results_dict[design_1]
                result_2 = results_dict[design_2]
                # pair is (design_1, design_2)
                # data_1 = result_1.data.clone()
                # data_2 = result_2.data.clone()

                # data_1 = result_1.data.cpu()
                # data_2 = result_2.data.cpu()

                data_1 = result_1.data
                data_2 = result_2.data

                pair_list.append((data_1, data_2))

                # pair_list.append((result_1.data, result_2.data))

        quadratic_num = len(results_dict) * len(results_dict)
        assert len(pair_list) == quadratic_num  # quadratic number of pairs

        # torch.cuda.empty_cache()

        # Turn pair_list into data_list satisfying the format defined by model forward.
        data_list = pair_list
        # for p in pair_list:
        #     data_list.append((p[0], p[1]))
        assert len(data_list) == quadratic_num

        saver.log_info(
            f'Sending {len(pair_list)}(={len(results_dict)}*{len(results_dict)}) design pairs into the model for comparison; len(data_list)={len(data_list)}'
        )

        test_loader = get_pairwise_data_loader(
            data_list, 'test', torch_generator, 'all'
        )

        probs = self.model_wrapper_pairwise.test(
            test_loader,
            data_list,
            self.config['evaluate'],
            mode='regression',
            forward_pairwise=True,
        )

        # saver.log_info(f'pairwise_probs={probs}')
        assert probs.shape == (quadratic_num, 2)
        self.log.debug(f'pairwise probs[0:10]={probs[0:10]}')

        if FLAGS.how_to_take_final == 'score_sorting':
            return self._score_sorting(designs_sorted, probs, results_dict, FLAGS.num_top_designs)
        elif FLAGS.how_to_take_final == 'ranked_choice_voting':

            return self._ranked_choice_voting(designs_sorted, probs, results_dict, FLAGS.num_top_designs)
        else:
            raise NotImplementedError()


    def _score_sorting(self, designs_sorted, probs, results_dict, K):
        # Initialize a dictionary to keep track of the scores
        scores = {design: 0.0 for design in results_dict}

        # Number of designs
        N = len(results_dict)
        design_names = designs_sorted

        # Calculate the scores for each design
        for i, design_1 in enumerate(design_names):
            for j, design_2 in enumerate(design_names):
                # Row index in the probs matrix for the (design_1, design_2) pair
                index = i * N + j
                scores[design_1] += probs[index][0].item()
                scores[design_2] += probs[index][1].item()

        self.log.debug(
            f'pairwise probs --> scores (random 10 entries)={random.sample(scores.items(), min(len(scores), 10))}'
        )

        sorted_scores = sorted(scores.values())
        self.log.debug(
            f"sorted scores (len={len(sorted_scores)}): smallest 10: {sorted_scores[:10]}, largest 10: {sorted_scores[-10:]}"
        )

        # Find the top 10 designs using a heap
        top_designs = heapq.nlargest(
            min(FLAGS.num_top_designs, N), scores.items(), key=lambda x: x[1]
        )

        return {design: results_dict[design] for design, _ in top_designs}


    def _ranked_choice_voting(self, designs_sorted, probs, results_dict, K):
        N = len(designs_sorted)
        remain = set(designs_sorted)
        removed_designs = set()

        while len(remain) > K:
            scores = {design: 0.0 for design in remain}

            for i, design_1 in enumerate(designs_sorted):
                if design_1 not in remain:
                    continue
                for j, design_2 in enumerate(designs_sorted):
                    if design_2 not in remain:
                        continue
                    index = i * N + j
                    scores[design_1] += probs[index][0].item()
                    scores[design_2] += probs[index][1].item()

            min_score = float('inf')
            worst_design = None
            for design, score in scores.items():
                if score < min_score:
                    min_score = score
                    worst_design = design
            assert worst_design is not None

            remain.remove(worst_design)
            removed_designs.add(worst_design)

        self.log.info(
            f'RCV: len(removed_designs)={len(removed_designs)}'
        )

        assert len(remain) == K
        return {design: results_dict[design] for design in remain}

    def _should_continue(self, timer):
        cond_time = (time.time() - timer) < self.timeout
        cond_point = self.explored_point < FLAGS.max_num_explored_points

        return cond_time and cond_point, cond_time, cond_point


class ModelInfoCollector:
    def __init__(self):
        self.time_in_sec_li = defaultdict(list)

    def collect_running_time(self, time_in_sec, model_type):
        self.time_in_sec_li[model_type].append(time_in_sec)

    def print_stats(self):
        for model_type, li in self.time_in_sec_li.items():
            print_stats(
                li,
                f'ModelInfoCollector: time_in_sec_li for model {model_type}',
                saver=saver,
            )


def get_top_k_results(best_results_dict, K):
    """
    Get the top K entries with the largest result.perf values from best_results_dict.

    Args:
        best_results_dict (dict): A dictionary mapping (perf, key) tuples to result objects.
        K (int): The number of top entries to retrieve.

    Returns:
        dict: A new dictionary containing the top K entries with the largest result.perf values.

    Description:
        This function creates a new dictionary that contains the top K entries from best_results_dict
        based on the result.perf values. It uses the heapq module to efficiently retrieve the top K
        entries without sorting the entire dictionary.

        If K is negative or 0, an empty dictionary is returned.
        If K is greater than the length of best_results_dict, all entries are returned.
        If K is within the valid range (1 to len(best_results_dict)), the top K entries are returned.

    Example:
        >>> best_results_dict = {
        ...     (0.9, 'key1'): result1,
        ...     (0.8, 'key2'): result2,
        ...     (0.7, 'key3'): result3,
        ...     (0.6, 'key4'): result4
        ... }
        >>> top_3_results = get_top_k_results(best_results_dict, 3)
        >>> top_3_results
        {
            (0.9, 'key1'): result1,
            (0.8, 'key2'): result2,
            (0.7, 'key3'): result3
        }
    """
    # Check edge cases
    if K <= 0:
        return {}
    if K >= len(best_results_dict):
        return best_results_dict

    # Use heapq to retrieve the top K entries efficiently
    top_k_entries = heapq.nlargest(K, best_results_dict.items(), key=lambda x: x[0][0])

    # Create a new dictionary with the top K entries
    top_k_results = {entry[0]: entry[1] for entry in top_k_entries}

    return top_k_results


