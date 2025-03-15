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

# from localdse.explorer.single_query import run_query
import pickle
import redis
from os.path import join
import argparse
from glob import iglob
import subprocess
import json

from utils import get_ts, create_dir_if_not_exists, save
from utils import get_src_path, extract_config_code
from tensorboardX import SummaryWriter
import time

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
            f'run_tool_{kernel}_{get_ts()}')
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

    parser = argparse.ArgumentParser(description='Running Queries')
    parser.add_argument('--kernel',
                        required=True,
                        action='store',
                        help='Kernel Name')
    parser.add_argument('--benchmark',
                        required=True,
                        action='store',
                        help='Benchmark Name')
    parser.add_argument('--root-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='GNN Root Directory')

    return parser.parse_args()
    
args = kernel_parser()
saver = Saver(args.kernel)
CHECK_EARLY_REJECT = False

src_dir = join(args.root_dir, 'dse_database/save/merlin_prj', args.kernel, 'xilinx_dse')
work_dir = join(args.root_dir, 'dse_database/save/merlin_prj', args.kernel, 'work_dir')
f_config = join(args.root_dir, 'dse_database', args.benchmark, 'config', f'{args.kernel}_ds_config.json')
f_pickle = join(args.root_dir, 'dse_database/save/merlin_prj', f'{args.kernel}.pickle')
db_dir = join(args.root_dir, 'dse_database', args.benchmark, 'databases', '**')
f_db = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result.db' in f and 'merged' in f][0]
print(f_db)
result_dict = pickle.load(open(f_pickle, "rb" ))

database = redis.StrictRedis(host='localhost', port=6379)
database.flushdb()
data = pickle.load(open(f_db, 'rb'))
database.hmset(0, data)
for _, result in sorted(result_dict.items()):
    key = f'lv2:{gen_key_from_design_point(result.point)}'
    lv1_key = key.replace('lv2', 'lv1')
    isEarlyRejected = False
    rerun = False
    if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):
        pickle_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickle_obj)
        if obj.ret_code.name == 'EARLY_REJECT':
            isEarlyRejected = True
    
    if database.hexists(0, key):
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        if obj.perf == 0.0:
            rerun = True

    if rerun or (not isEarlyRejected and not database.hexists(0, key)):
        kernel = args.kernel
        print(f_config)
        with open(f'./localdse/kernel_results/{args.kernel}_point.pickle', 'wb') as handle:
            pickle.dump(result.point, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # p = subprocess.Popen(f'cd {get_src_path()} \n source ~/env.sh ', shell = True, stdout=subprocess.PIPE)
        p = subprocess.Popen(f'cd {get_src_path()} \n source ~/env.sh \n ~/merlin_docker/docker-run.sh python3.6 -m localdse.explorer.single_query --src-dir {src_dir} --work-dir {work_dir} --kernel {kernel} --config {f_config}', shell = True, stdout=subprocess.PIPE)
        p.wait()
        text = (p.communicate()[0]).decode('utf-8')
        saver.debug('############################')
        saver.debug(f'Recieved output for {key}')
        saver.debug(text)
        saver.debug('############################')

        q_result = pickle.load(open(f'localdse/kernel_results/{args.kernel}.pickle', 'rb'))
        saver.info(q_result)
        
        # q_result = run_query(result.point, src_dir, work_dir, args.kernel, f_config)
        for key, result in q_result.items():
            pickled_result = pickle.dumps(result)
            database.hset(0, key, pickled_result)
            saver.info(f'Performance for {key}: {result.perf} with return code: {result.ret_code}')
        if 'EARLY_REJECT' in text:
            for key, result in q_result.items():
                if result.ret_code != 'EARLY_REJECT':
                    result.ret_code = 'EARLY_REJECT'
                    result.perf = 0.0
                    pickled_result = pickle.dumps(result)
                    database.hset(0, key.replace('lv2', 'lv1'), pickled_result)
                    #saver.info(f'Performance for {key}: {result.perf}')
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
        saver.info(f'Key exists. Performance for {key}: {result.actual_perf} with return code: {result.ret_code}')

