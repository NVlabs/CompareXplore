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

from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Union


DesignPoint = Dict[str, Union[int, str]]

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
