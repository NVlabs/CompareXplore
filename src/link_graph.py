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

from utils import get_root_path
from saver import saver

from os.path import basename
from pathlib import Path

def find_ast(programl_gexf_path):
    bn = basename(programl_gexf_path)
    source_path = Path(f'{get_root_path()}/dse_database/machsuite/dot-files/original-size')
    # source_path = Path(f'{get_root_path()}/dse_database/machsuite/sources/original-size')
    source_path_polly = Path(f'{get_root_path()}/dse_database/polly/dot-files')

    if 'gemm-blocked' in bn:
        return Path(f'{source_path}/gemm-blocked_kernel.c.gexf')
    elif 'gemm-ncubed' in bn:
        return Path(f'{source_path}/gemm-ncubed_kernel.c.gexf')
    elif 'stencil_stencil2d' in bn:
        return Path(f'{source_path}/stencil_kernel.c.gexf')
    elif 'aes' in bn:
        return Path(f'{source_path}/aes_kernel.c.gexf')
    elif 'nw' in bn:
        return Path(f'{source_path}/nw_kernel.c.gexf')
    elif 'spmv-crs' in bn:
        return Path(f'{source_path}/spmv-crs_kernel.c.gexf')
    elif 'spmv-ellpack' in bn:
        return Path(f'{source_path}/spmv-ellpack_kernel.c.gexf')
    elif 'atax' in bn:
        return Path(f'{source_path_polly}/atax_kernel.c.gexf')
    elif 'mvt' in bn:
        return Path(f'{source_path_polly}/mvt_kernel.c.gexf')
    else:
        saver.log_info(f'Cannot find ast gexf for {programl_gexf_path}')
        return None