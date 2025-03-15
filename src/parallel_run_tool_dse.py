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

# from localdse.explorer.single_query import run_query
import pickle
import redis
from os.path import join, dirname
import argparse
from glob import iglob
import subprocess
import json
from copy import deepcopy
import shutil

from utils import get_ts, create_dir_if_not_exists, get_src_path
from tensorboardX import SummaryWriter
import time
from subprocess import Popen, DEVNULL, PIPE

class MyTimer():
    def __init__(self) -> None:
        self.start = time.time()
    
    def elapsed_time(self):
        end = time.time()
        minutes, seconds = divmod(end - self.start, 60)
        
        return int(minutes)

class Saver():
    def __init__(self, kernel):
        self.logdir = join(
            get_src_path(),
            'logs',
            # f'MAML-wo-zscore-run_tool-class-off_{kernel}_{get_ts()}')
            # 'yunsheng', 'dac-short', 'yizhou-spread-ds', f'MAML9-20d-5t-wo-zscore-run_tool-class-off_{kernel}_{get_ts()}')
            # 'yunsheng', 'dac-short', 'gnn-dse', f'spread-ds-run_tool-class-on_{kernel}_{get_ts()}')
            # 'yunsheng', 'm-post-dac', 'sepPT', f'kmeansNN-10d-run_tool-class-on_{kernel}_{get_ts()}')
            'post-dac', 'sa', f'pure-model-wo-zscore-run_tool-class-off_{kernel}_{get_ts()}')
            #f'MAML-run_tool-case1a_genetic-up3-class-off-near200_{kernel}_{get_ts()}')
        create_dir_if_not_exists(self.logdir)
        self.writer = SummaryWriter(self.logdir)
        self.timer = MyTimer()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')
    
    def info(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] INFO: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] INFO: {s}\n')
        self.log_f.flush()
        
    def error(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] ERROR: {s}')
        if not hasattr(self, 'log_e'):
            self.log_e = self._open('error.txt')
        self.log_e.write(f'[{elapsed}m] ERROR: {s}\n')
        self.log_e.flush()
        
    def warning(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] WARNING: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] WARNING: {s}\n')
        self.log_f.flush()
        
    def debug(self, s, silent=True):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] DEBUG: {s}')
        if not hasattr(self, 'log_d'):
            self.log_d = self._open('debug.txt')
        self.log_d.write(f'[{elapsed}m] DEBUG: {s}\n')
        self.log_d.flush()

def gen_key_from_design_point(point) -> str:

    return '.'.join([
        '{0}-{1}'.format(pid,
                         str(point[pid]) if point[pid] else 'NA') for pid in sorted(point.keys())
    ])

def kernel_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser_run = argparse.ArgumentParser(description='Running Queries')
    parser_run.add_argument('--kernel',
                        required=True,
                        action='store',
                        help='Kernel Name')
    parser_run.add_argument('--benchmark',
                        required=True,
                        action='store',
                        help='Benchmark Name')
    parser_run.add_argument('--root-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='GNN Root Directory')
    parser_run.add_argument('--redis-port',
                        required=True,
                        action='store',
                        default='6379',
                        help='The port number for redis database')

    return parser_run.parse_args()
    
def persist(database, db_file_path) -> bool:
    #pylint:disable=missing-docstring

    dump_db = {
        key: database.hget(0, key)
        for key in database.hgetall(0)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True

def run_procs(saver, procs, database, kernel, f_db_new):
    saver.info(f'Launching a batch with {len(procs)} jobs')
    try:
        while procs:
            prev_procs = list(procs)
            procs = []
            for p_list in prev_procs:
                text = 'None'
                # print(p_list)
                idx, key, p = p_list
                # text = (p.communicate()[0]).decode('utf-8')
                ret = p.poll()
                # Finished and unsuccessful
                if ret is not None and ret != 0:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.info(f'Job with batch id {idx} has non-zero exit code: {ret}')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')
                # Finished and successful
                elif ret is not None:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')

                    q_result = pickle.load(open(f'localdse/kernel_results/{kernel}_{idx}.pickle', 'rb'))

                    for _key, result in q_result.items():
                        pickled_result = pickle.dumps(result)
                        if 'lv2' in key:
                            database.hset(0, _key, pickled_result)
                        saver.info(f'Performance for {_key}: {result.perf} with return code: {result.ret_code} and resource utilization: {result.res_util}')
                    if 'EARLY_REJECT' in text:
                        for _key, result in q_result.items():
                            if result.ret_code != 'EARLY_REJECT':
                                result.ret_code = 'EARLY_REJECT'
                                result.perf = 0.0
                                pickled_result = pickle.dumps(result)
                                database.hset(0, _key.replace('lv2', 'lv1'), pickled_result)
                                #saver.info(f'Performance for {key}: {result.perf}')
                    persist(database, f_db_new)
                # Still running
                else:
                    procs.append([idx, key, p])
                
                time.sleep(1)
    except:
        saver.error(f'Failed to finish the processes')
        raise RuntimeError()


args = kernel_parser()
saver = Saver(args.kernel)
CHECK_EARLY_REJECT = False

src_dir = join(args.root_dir, 'dse_database/save/merlin_prj', f'{args.kernel}', 'xilinx_dse')
# work_dir = join(args.root_dir, 'dse_database/save/merlin_prj', f'{args.kernel}', 'work_dir')
work_dir = join('/expr', f'{args.kernel}', 'work_dir')
f_config = join(args.root_dir, 'dse_database', args.benchmark, 'config', f'{args.kernel}_ds_config.json')
# f_pickle = join(args.root_dir, 'dse_database/timeouts', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_task-transfer/fine-tune/', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_task-transfer/fine-tune-oldspeed-fromv1-todse4/', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_task-transfer/fine-tune-dse4-fromv1/', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_task-transfer/fine-tune-dse4-fromv1-freeze5/', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_task-transfer/fine-tune-dse4-only-new-points-tune-all/', f'{args.kernel}.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/maml/', f'2mm-10t-42d-maml-regular.pickle')
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/maml/', f'2mm-newdb-MAML-64d-5t.pickle') # maml/2mm-newdb-64d-2048t.pickle
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_1_gemm-p_1_20designs_5times_MAML_1.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_8_gemm-p_8_10designs_5times_MAML_2.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_3_3mm_3_10designs_5times_MAML_5.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_59_jacobi-1d_19_10designs_5times_MAML_2.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_19_3mm_19_50designs_5times_MAML_5.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_19_gemver_19_10designs_5times_MAML_2.pickle') 
#  f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/', f'final_test_19_gemver_19_20designs_5times_MAML_3.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/maml-aes', f'final_test_19_3mm_19_10designs_5times_MAML_1.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/maml-aes', f'final_test_39_fdtd-2d_19_10designs_5times_MAML_2.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/maml-aes', f'final_test_59_jacobi-1d_19_10designs_5times_MAML_1.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short', f'final_test_39_fdtd-2d_19_50designs_5times_MAML_2.pickle') 
# f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short', f'final_test_19_3mm_19_20designs_5times_MAML_6.pickle') 
# f_pickle_path = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/dac-short/yizhou-sampling-spread-ds', '**') 
# f_pickle_path = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_yunsheng/post-dac-short/', '**/*') 
# f_pickle_path = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_gnn-dse/spread-ds', '**') 
f_pickle_path = join(args.root_dir, 'dse_database/save/merlin_prj/pickles_gnn-dse/unadapted/sa', '**') 
f_pickle_list = [f for f in iglob(f_pickle_path, recursive=True) if f.endswith('.pickle') and args.kernel in f]
assert len(f_pickle_list) == 1
f_pickle = f_pickle_list[0]
db_dir = join(args.root_dir, 'dse_database', args.benchmark, 'databases', '**')
result_dict = pickle.load(open(f_pickle, "rb" ))
create_dir_if_not_exists(dirname(work_dir))
create_dir_if_not_exists(work_dir)

max_db_id = 15
min_db_id = -1
found_db = False
for i in range(max_db_id, min_db_id, -1):
    # f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-{i}.db' in f and 'v20' in f and ('single-merged' in f or 'transfer' in f)]
    # f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-{i}.db' in f and ('single-merged' in f or 'transfer' in f)]
    # f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-yunsheng-maml-{i}.db' in f and ('single-merged' in f or 'transfer' in f)]
    # f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-yunsheng-dac-short-{i}.db' in f and ('single-merged' in f or 'transfer' in f)]
    # f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-yunsheng-dac-short-{i}.db' in f and ('single-merged' in f or 'transfer' in f)]
    f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_new_updated-gnndse-dac-short-{i}.db' in f and ('single-merged' in f or 'transfer' in f)]
    if len(f_db_list) == 1:
        f_db = f_db_list[0]
        print(f_db)
        # f_db_new = f_db.replace(f'_updated-{i}', f'_updated-yunsheng-maml-{i}')
        # f_db_new = f_db.replace(f'_updated-yunsheng-maml-{i}', f'_updated-yunsheng-maml-{i+1}')
        f_db_new = f_db.replace(f'_updated-gnndse-dac-short-{i}', f'_updated-gnndse-dac-short-{i+1}')
        # f_db_new = f_db.replace(f'_updated-yunsheng-dac-short-{i}', f'_updated-yunsheng-dac-short-{i+1}')
        # f_db_new = f_db.replace(f'_updated-yunsheng-dac-short-{i}', f'_updated-yunsheng-post-dac-short-{i}')
        # f_db_new = f_db.replace(f'_updated-{i}', f'_updated-yunsheng-dac-short-{i+1}')
        found_db = True
        break

if not found_db:
    f_db_list = [f for f in iglob(db_dir, recursive=True) if f.endswith('.db') and 'bottleneck' in f and f'{args.kernel}_result_' in f and 'v20' in f]
    if len(f_db_list) == 1:
        f_db = f_db_list[0]
        print(f_db)
        # f_db_new = f_db.replace(f'_result', f'_result_updated1_updated')    
        f_db_new = join(dirname(f_pickle), f'{args.kernel}.db')    

    
# try:
#     # f_db = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated.db' in f and 'merged' in f][0]
#     f_db = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated7_updated.db' in f and 'merged' in f][0]
#     # f_db = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result.db' in f and 'merged' in f][0]
#     print(f_db)
    
#     # f_db_new = f_db.replace('_result', '_result_updated2')
#     f_db_new = f_db.replace('_updated7', '_updated8')
# except:
#     f_db = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated3_updated.db' in f and 'merged' in f][0]
#     print(f_db)
#     f_db_new = f_db.replace('_updated3', '_updated4')

database = redis.StrictRedis(host='localhost', port=int(args.redis_port))
database.flushdb()
try:
    file_db = open(f_db, 'rb')
    data = pickle.load(file_db)
    database.hmset(0, data)
except:
    f_db_new = f'{args.kernel}_result_updated{1}_updated.db'
    saver.info('No prior databases')

# batch_num = len(result_dict)
batch_num = 5
batch_id = 0
procs = []
saver.info(f"""processing {f_pickle} 
    from db: {f_db} and 
    updating to {f_db_new}""")
saver.info(f"total of {len(result_dict.keys())} solution(s)")
for _, result in sorted(result_dict.items()):
    if len(procs) == batch_num:
        run_procs(saver, procs, database, args.kernel, f_db_new)
        batch_id == 0
        procs = []
    for key_, value in result.point.items():
        if type(value) is str:
            result.point[key_] = value
        else:
            result.point[key_] = value.item()
    key = f'lv2:{gen_key_from_design_point(result.point)}'
    # print(key)
    lv1_key = key.replace('lv2', 'lv1')
    isEarlyRejected = False
    rerun = False
    if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):
        pickle_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickle_obj)
        if obj.ret_code.name == 'EARLY_REJECT':
            isEarlyRejected = True
    
    if database.hexists(0, key):
        # print(f'key exists {key}')
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        if obj.perf == 0.0:
            # print(f'should rerun for {key}')
            rerun = True

    if rerun or (not isEarlyRejected and not database.hexists(0, key)):
        kernel = args.kernel
        # print(f_config)
        # point = deepcopy(result.point)
        point = result.point
        # print(point)
        # for key_, value in result.point.items():
        #     if type(value) is str:
        #         point[key_] = value
        #     else:
        #         point[key_] = value.item()
        with open(f'./localdse/kernel_results/{args.kernel}_point_{batch_id}.pickle', 'wb') as handle:
           pickle.dump(point, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'./localdse/kernel_results/{args.kernel}_point_{batch_id}.json', 'w') as f:
        #     json.dump(point, f)
        new_work_dir = join(work_dir, f'batch_id_{batch_id}')
        # shutil.rmtree(new_work_dir)
        # create_dir_if_not_exists(new_work_dir)
        # p = Popen(f'cd {get_src_path()} \n source /curr/atefehSZ/env.sh \n /curr/atefehSZ/merlin_docker/docker-run.sh -i /bin/bash \n faketime "2020-12-24 08:15:42" python3.6 -m localdse.explorer.single_query --src-dir {src_dir} --work-dir {new_work_dir} --kernel {kernel} --config {f_config} --id {batch_id}', shell = True, stdout=PIPE)
        # p = Popen(f'cd {get_src_path()} \n source /curr/atefehSZ/env.sh \n /curr/atefehSZ/merlin_docker/docker-run-gnn.sh single {src_dir} {new_work_dir} {kernel} {f_config} {batch_id}', shell = True, stdout=PIPE)
        # p = Popen(f'cd {get_src_path()} \n source /curr/atefehSZ/vitis_env.sh \n /curr/atefehSZ/merlin_docker/vitis_docker-run-gnn.sh single {src_dir} {new_work_dir} {kernel} {f_config} {batch_id}', shell = True, stdout=PIPE, stderr=PIPE)
        # print(f'/share/atefehSZ/merlin_docker/docker-run-gnn.sh single {src_dir} {new_work_dir} {kernel} {f_config} {batch_id}')
        p = Popen(f'cd {get_src_path()} \n source /share/atefehSZ/env.sh \n /share/atefehSZ/merlin_docker/docker-run-gnn.sh single {src_dir} {new_work_dir} {kernel} {f_config} {batch_id}', shell = True, stdout=PIPE)
        

        # if batch_id < batch_num: 
        procs.append([batch_id, key, p])
        saver.info(f'Added {result.point} with batch id {batch_id}')
        batch_id += 1
    elif isEarlyRejected:
        pickled_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickled_obj)
        result.actual_perf = 0
        result.ret_code = 'EARLY_REJECT'
        result.valid = False
        saver.info(f'LV1 Key exists for {key}, EARLY_REJECT')
    else:
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        result.actual_perf = obj.perf
        saver.info(f'Key exists. Performance for {key}: {result.actual_perf} with return code: {result.ret_code} and resource utilization: {obj.res_util}')

if len(procs) > 0:
    run_procs(saver, procs, database, args.kernel, f_db_new)
            
    

try:
    file_db.close()
except:
    print('file_db is not defined')

