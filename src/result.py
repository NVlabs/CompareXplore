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

"""
The definition of evaluation results
"""

from parameter import DesignPoint

from torch_geometric.data import Data
from enum import Enum
from typing import Dict, List, NamedTuple, Optional
import pickle
import numpy as np
from collections import Counter


def persist(database, db_file_path) -> bool:
    # pylint:disable=missing-docstring

    dump_db = {
        key: database.hget(0, key)
        for key in database.hgetall(0)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True


class Job(object):
    """The info and properties of a job"""

    class Status(Enum):
        INIT = 0
        APPLIED = 1

    def __init__(self, path: str):
        self.path: str = path
        self.key: str = 'NotAPPLIED'
        self.point: Optional[DesignPoint] = None
        self.status: Job.Status = Job.Status.INIT


class Result(object):
    """The base module of evaluation result"""

    class RetCode(Enum):
        PASS = 0
        UNAVAILABLE = -1
        ANALYZE_ERROR = -2
        EARLY_REJECT = -3
        TIMEOUT = -4
        DUPLICATED = -5

    def __init__(self, ret_code_str: str = 'PASS'):

        # The design point of this result.
        self.point: Optional[DesignPoint] = None

        # Also store the encoded data so that the verification stage can reuse the Data object to run pairwise comparison.
        self.data: Data = None

        # The return code of the evaluation
        self.ret_code: Result.RetCode = self.RetCode[ret_code_str]

        # Indicate if this result is valid to be a final output. For example, a result that
        # out-of-resource is invalid.
        self.valid: bool = False

        # The job path for this result (if available)
        self.path: Optional[str] = None

        # The quantified QoR value. Larger the better.
        self.quality: float = -float('inf')

        # Performance in terms of estimated cycle or onboard runtime.
        self.perf: float = 0.0

        # Resource utilizations
        self.res_util: Dict[str, float] = {
            'util-BRAM': 0,
            'util-DSP': 0,
            'util-LUT': 0,
            'util-FF': 0,
            'total-BRAM': 0,
            'total-DSP': 0,
            'total-LUT': 0,
            'total-FF': 0
        }

        # Elapsed time for evaluation
        self.eval_time: float = 0.0

    @staticmethod
    def aggregate_results(results):
        assert type(results) is list and len(results) >= 1
        rtn = Result()

        point = None
        data = None
        for r in results:
            if point is None:
                point = r.point
                data = r.data
            else:
                assert point == r.point, f'Results to aggregate must contain the same point'
        rtn.point = point
        rtn.data = data

        perf_li = []
        actual_perf_li = []
        quality_li = []
        valid_li = []
        for r in results:
            perf_li.append(r.perf)
            actual_perf_li.append(r.actual_perf)
            quality_li.append(r.quality)
            assert r.valid in [True, False]
            valid_li.append(r.valid)
         # average the prediction
        rtn.perf = np.mean(perf_li)
        rtn.actual_perf = np.mean(actual_perf_li)
        rtn.quality = np.mean(quality_li)
        rtn.valid = Result.majority_voting(valid_li) # tricky: a boolean list

        for key in rtn.res_util.keys():
            value_list = []
            for r in results:
                value_list.append(r.res_util[key])
            rtn.res_util[key] = np.mean(value_list)

        return rtn
    
    @staticmethod
    def majority_voting(x):
        c = Counter(x)
        potential_ties = []
        for k, v in c.most_common():
            potential_ties.append((v, k))
        rtn = sorted(potential_ties)[-1][1]
        return rtn
    

class MerlinResult(Result):
    """The result after running Merlin transformations"""

    def __init__(self, ret_code_str: str = 'PASS'):
        super(MerlinResult, self).__init__(ret_code_str)

        # Critical messages from the Merlin transformations
        self.criticals: List[str] = []

        # The kernel code hash for recognizing duplications
        self.code_hash: Optional[str] = None


class HierPathNode(NamedTuple):
    """The datastructure of hierarchy path node"""
    nid: str
    latency: float
    is_compute_bound: bool


class HLSResult(Result):
    """The result after running the HLS"""

    def __init__(self, ret_code_str: str = 'PASS'):
        super(HLSResult, self).__init__(ret_code_str)

        # A list of hierarchy paths in the order of importance
        self.ordered_paths: Optional[List[List[HierPathNode]]] = None


class BitgenResult(Result):
    """The result after bit-stream generation"""

    def __init__(self, ret_code_str: str = 'PASS'):
        super(BitgenResult, self).__init__(ret_code_str)

        # Frequency
        self.freq: float = 0.0

