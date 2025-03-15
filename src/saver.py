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
from utils import get_ts, create_dir_if_not_exists, save, save_pickle
from utils import (
    get_src_path,
    get_model_info_as_str,
    extract_config_code,
    plot_scatter_line,
    plot_dist,
    save_pickle,
    dirname,
    estimate_model_size,
    set_ts,
    print_gpu_free_mem_info,
    get_gpu_info,
)
from tensorboardX import SummaryWriter


from collections import OrderedDict, defaultdict
from pprint import pprint
from os.path import join, dirname, basename
import torch
import networkx as nx
import numpy as np
import time
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


class MyTimer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed_time(self):
        end = time.time()
        minutes, seconds = divmod(end - self.start, 60)

        return int(minutes)


class Saver(object):
    def __init__(self):

        # model_str = self._get_model_str()

        # The following code is added for acclerator launch. Essentially launch.py will set the parameters below.
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("-ts", "--time_stamp", required=False)
        parser.add_argument("-bs", "--batch_size", required=False)
        parser.add_argument("-mode", "--mode", required=False)
        parser.add_argument("-DEVICES", "--DEVICES", required=False)
        parser.add_argument("-gpu_free_mem_info", "--gpu_free_mem_info", required=False)
        parser.add_argument("-launch_cmd", "--launch_cmd", required=False)

        args = parser.parse_args()
        if args.time_stamp is not None:
            set_ts(args.time_stamp)
        if args.batch_size is not None:
            setattr(FLAGS, 'batch_size', int(args.batch_size))
        if args.mode is not None:
            setattr(FLAGS, 'mode', args.mode)
        if args.DEVICES is not None:
            setattr(FLAGS, 'DEVICES', args.DEVICES)
            delattr(FLAGS, 'device')
            delattr(FLAGS, 'gpu')
        if args.gpu_free_mem_info is not None:
            self.gpu_free_mem_info = eval(args.gpu_free_mem_info)
        else:
            self.gpu_free_mem_info = get_gpu_info()
            # print(f'self.gpu_free_mem_info={self.gpu_free_mem_info}')

        if args.launch_cmd is not None:
            setattr(FLAGS, 'launch_cmd', args.launch_cmd)

        subtask_str = '_'.join([s[0:2] for s in FLAGS.subtask.split('_')])
        self.logdir = join(
            get_src_path(),
            'logs',
            # '{}_{}_{}_{}_{}_{}_{}'.format(FLAGS.norm_method, FLAGS.task, FLAGS.subtask, FLAGS.tag, FLAGS.target, model_str, get_ts()))
            '{}_{}_{}_{}_{}'.format(
                subtask_str, get_ts(), FLAGS.task[0:1], FLAGS.hostname, FLAGS.user
            ),
        )

        if FLAGS.pairwise_class:
            # https://github.com/huggingface/accelerate/issues/24
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            kwargs_handlers = [ddp_kwargs]
        else:
            kwargs_handlers = []
        print(f'kwargs_handlers={kwargs_handlers}')

        if FLAGS.mode == 'acc_launch':
            self.accelerator = Accelerator(
                gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
                kwargs_handlers=kwargs_handlers,
            )
        else:
            self.accelerator = None

        if FLAGS.mode == 'acc_launch':
            pid = self.accelerator.process_index
            setattr(FLAGS, 'pid', pid)
            self.pidstr = f'_{pid}'
        else:
            self.pidstr = ''

        if FLAGS.mode == 'acc_launch':
            context = self.accelerator.main_process_first()
        else:
            context = NoOpContextManager()

        with context:
            create_dir_if_not_exists(self.logdir)
            self.writer = SummaryWriter(join(self.logdir, 'runs'))
            self.model_info_f = self._open(f'model_info{self.pidstr}.txt')
            self.plotdir = join(self.logdir, 'plot')
            create_dir_if_not_exists(self.plotdir)
            self.objdir = join(self.logdir, 'obj')
            self._log_model_info()
            self._save_conf_code()
            self.timer = MyTimer()
            self.all_messages = set()
            self.msg_type_count = defaultdict(int)
            print('Logging to:\n{}'.format(self.logdir))
            print(basename(self.logdir))

        print_gpu_free_mem_info(self.gpu_free_mem_info, self)

    def get_device(self):
        if FLAGS.mode == 'acc_launch':
            pid = self.accelerator.process_index
            which_gpu = pid
            saver.log_info(f'get_device: pid={pid}; which_gpu={which_gpu}')
            return f'cuda:{which_gpu}'
        else:
            return FLAGS.device

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.log_info(self.logdir)
        self.log_info(basename(self.logdir))
        self.writer.close()
        if hasattr(self, 'log_f'):
            self.log_f.close()
        if hasattr(self, 'log_e'):
            self.log_e.close()
        if hasattr(self, 'log_d'):
            self.log_d.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_log_dir(self):
        return self.logdir

    def get_plot_dir(self):
        create_dir_if_not_exists(self.plotdir)
        return self.plotdir

    def get_obj_dir(self):
        create_dir_if_not_exists(self.objdir)
        return self.objdir

    def log_list_of_lists_to_csv(self, lol, fn, delimiter=','):
        import csv

        fp = open(join(self.logdir, fn), 'w+')
        csv_writer = csv.writer(fp, delimiter=delimiter)
        for l in lol:
            csv_writer.writerow(l)
        fp.close()

    def log_dict_of_dicts_to_csv(self, fn, csv_dict, csv_header, delimiter=','):
        import csv

        fp = open(join(self.logdir, f'{fn}.csv'), 'w+')
        f_writer = csv.DictWriter(fp, fieldnames=csv_header)
        f_writer.writeheader()
        for d, value in csv_dict.items():
            if d == 'header':
                continue
            f_writer.writerow(value)
        fp.close()

    def save_emb_accumulate_emb(self, data_key, d, convert_to_np=False):
        if not hasattr(self, 'emb_dict'):
            self.emb_dict = OrderedDict()

        if convert_to_np:
            new_d = {}
            for key, val in d.items():
                if torch.is_tensor(val):
                    val = val.detach().cpu().numpy()
                new_d[key] = val
            d = new_d

        self.emb_dict[data_key] = d

    def save_emb_save_to_disk(self, p):
        assert hasattr(self, 'emb_dict')
        filepath = join(self.objdir, p)
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(self.emb_dict, filepath, print_msg=True)

    def save_emb_dict(self, d, p, convert_to_np=False):
        if not hasattr(self, 'save_emb_cnt'):
            self.save_emb_cnt = 0

        if convert_to_np:
            new_d = {}
            for key, val in d.items():
                if torch.is_tensor(val):
                    val = val.detach().cpu().numpy()
                new_d[key] = val
            d = new_d

        filepath = join(self.objdir, f'{self.save_emb_cnt}_{p}')
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(d, filepath, print_msg=True)
        self.save_emb_cnt += 1

    def log_dict_to_json(self, dictionary, fn):
        import json

        # as requested in comment
        with open(join(self.get_obj_dir(), fn), 'w') as file:
            file.write(json.dumps(dictionary))

    def log_model_architecture(self, model):
        # print(model)
        if hasattr(self, 'has_logged_arch') and self.has_logged_arch:
            return  # avoid logging again and again within one python process
        if not hasattr(self, 'has_logged_arch'):
            self.has_logged_arch = True
        self.model_info_f.write('{}\n'.format(model))
        estimate_model_size(model, 'whole model', self)
        self.model_info_f.flush()
        # self.model_info_f.close()  # TODO: check in future if we write more to it

    def log_info(self, s, silent=False, build_str=None):
        if not silent:
            print(s)
        if not hasattr(self, 'log_f'):
            self.log_f = self._open(f'log{self.pidstr}.txt')
        try:
            self.log_f.write('{}\n'.format(s))
            self.log_f.flush()
        except:
            raise RuntimeError()
        if build_str is not None:
            assert type(build_str) is str
            build_str += '{}\n'.format(s)
            return build_str

    def log_info_once(self, s, silent=False):
        if s not in self.all_messages:
            self.all_messages.add(s)
            self.log_info(s, silent)

    def log_info_at_most(self, s, msg_type, times, silent=False):
        if self.msg_type_count[msg_type] < times:
            self.log_info(s, silent)
            self.msg_type_count[msg_type] += 1

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

    def log_info_new_file(self, s, fn):
        # print(s)
        log_f = open(join(self.logdir, fn), 'a')
        log_f.write('{}\n'.format(s))
        log_f.close()

    def save_dict(self, d, p, subfolder=''):
        filepath = join(self.logdir, subfolder, p)
        create_dir_if_not_exists(dirname(filepath))
        save_pickle(d, filepath, print_msg=True)
        # print(f'dict of keys {d.keys()} saved to {filepath}')

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        save({'FLAGS': FLAGS}, p, print_msg=False)

    # def save_flags(self, fn):
    #     p = join(self.get_log_dir(), fn)
    #     save({'FLAGS': FLAGS}, p, print_msg=False)

    def save_trained_model(self, trained_model, ext='', path=None, info=None):
        if FLAGS.mode == 'standalone' or self.accelerator.is_main_process:
            # https://huggingface.co/docs/accelerate/concept_guides/deferring_execution

            model_dir = join(self.logdir, 'models')

            if FLAGS.mode == 'acc_launch':
                trained_model = self.accelerator.unwrap_model(trained_model)

            create_dir_if_not_exists(model_dir)
            if path is None:
                p = join(model_dir, 'trained_model{}.pt'.format(ext))
            else:
                p = path

            to_save = {'model_weights': trained_model.state_dict()}
            if info is not None:
                to_save['info'] = info
            torch.save(to_save, p)
            self.log_info('Trained model saved to {}'.format(p))

            # kname = 'bert_model.block.0.layer.0.SelfAttention.q.weight'
            # v = trained_model.state_dict().get(kname)
            # if v is not None:
            #     saver.log_info(f'(For reference, kname {kname}[0:2]: {v[0:2]}')

    def load_trained_model(self, model_path, model):
        print_info = None
        if model_path != None and model_path != 'None':
            if FLAGS.load_model_weights:
                saver.info(f'loading model from {model_path}')
                ld = torch.load(model_path, map_location=torch.device('cpu'))
                # ld.pop('bert_model.block.2.layer.0.SelfAttention.v.weight')
                add_info = ld.get('add_info')
                info = ld.get('info')
                # weights = ld
                if add_info is not None or info is not None:
                    if add_info is not None:
                        print_info = add_info
                    else:
                        print_info = info
                    saver.log_info(f'Info about the loaded model: {print_info}')
                    weights = ld['model_weights']
                else:
                    weights = ld.get('model_weights')
                    if weights is None:
                        weights = ld  # older code just stores weights

                strict = True

                # to_delete = False
                # if hasattr(model, 'pretrained_GNN_encoder'):
                #     pretrained_GNN_encoder = model.pretrained_GNN_encoder
                #     node_embs_proj_to_pretrained_model = model.node_embs_proj_to_pretrained_model
                #     delattr(model, "pretrained_GNN_encoder")
                #     delattr(model, "node_embs_proj_to_pretrained_model")
                #     to_delete = True
                #     strict = False

                if FLAGS.load_model_HARPnet:
                    strict = False

                weights_fixed = {}
                for key, value in weights.items():
                    if key.startswith('module.'):
                        weights_fixed[key.replace('module.', '')] = (
                            value  # for some reason, recently saved model weights have "module.bert_model.embed_tokens.weight" as attributes
                        )
                    else:
                        weights_fixed[key] = value
                weights = weights_fixed

                try:
                    model.load_state_dict(weights, strict=strict)
                except Exception as e:
                    self.log_info(f'Error in loading!\n{[t[0] for t in model.named_parameters()]}')
                    raise e

                # if to_delete:
                #     model.pretrained_GNN_encoder = pretrained_GNN_encoder
                #     model.node_embs_proj_to_pretrained_model = node_embs_proj_to_pretrained_model

                saver.info(
                    f'Model (loaded) is on device {next(model.parameters()).device}'
                )
            else:
                self.log_info(
                    f'FLAGS.load_model_weights={FLAGS.load_model_weights}: No loading; Random model'
                )

        if FLAGS.mode == "acc_launch":
            pass
        else:
            assert FLAGS.mode == "standalone" 
            model = model.to(FLAGS.device)

        return model, print_info

    def flatten_list_tuple_into_np_arr(self, li):
        li_flatten = []
        for elt in li:
            for elt_elt in elt:
                li_flatten.append(elt_elt)
        return np.array(li_flatten)

    def log_scatter_mcs(self, cur_id, iter, result_d):
        sp = join(self.get_obj_dir(), f'{iter}_val')
        save(result_d, sp)
        self._save_to_result_file(f'iter {iter} val result', to_print=True)
        for label, data_dict in result_d.items():
            if FLAGS.val_debug:
                g1, g2 = data_dict['g1'], data_dict['g2']
                nx.write_gexf(
                    g1, join(self.get_obj_dir(), f'{cur_id}_{g1.graph["gid"]}.gexf')
                )
                nx.write_gexf(
                    g2, join(self.get_obj_dir(), f'{cur_id}_{g2.graph["gid"]}.gexf')
                )
            plot_scatter_line(data_dict['result'], label, self.get_plot_dir())

        for label, data_dict in result_d.items():
            for model_name, data_dict_elt in data_dict['result'].items():
                incumbent_size_list = data_dict_elt['incumbent_data']
                runtime = data_dict_elt['runtime']
                self._save_to_result_file(
                    f'  num_iters={len(incumbent_size_list)}, '
                    f'runtime={runtime}, '
                    f'mcs={incumbent_size_list[-1]}, '
                    f'method={model_name}',
                    to_print=True,
                )

    def save_graph_as_gexf(self, g, fn):
        nx.write_gexf(g, join(self.get_obj_dir(), fn))

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    # def clean_up_saved_models(self, best_iter):
    #     for file in glob('{}/models/*'.format(self.get_log_dir())):
    #         if str(best_iter) not in file:
    #             system('rm -rf {}'.format(file))

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    def save_dict_as_pickle(self, d, name):
        p = join(self.get_obj_dir(), name)
        p = save(d, p, print_msg=True, use_klepto=False)
        self.log_info(f'Saving dict {name}')
        return p

    def _get_model_str(self):
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset]
        for f in key_flags:
            li.append(str(f))
        rtn = '_'.join(li)
        return f'{rtn}'

    def _log_model_info(self):
        s = get_model_info_as_str(FLAGS)
        print(s)
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.flush()
        # self.writer.add_text('model_info_str', s)

    def log_new_FLAGS_to_model_info(self):
        self.model_info_f.write('----- new model info after loading\n')
        self._log_model_info()
        self._save_conf_code()

    def _save_to_result_file(self, obj, name=None, to_print=False):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            if to_print:
                print(obj)
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))
        self.results_f.flush()

    def _shrink_space_pairs(self, pairs):
        for _, pair in pairs.items():
            # print(pair.__dict__)
            pair.shrink_space_for_save()
            # pass
            # print(pair.__dict__)
            # exit(-1)
        return pairs


class NoOpContextManager:
    """A no-operation context manager that does nothing."""
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False



saver = Saver()  # can be used by `from saver import saver`
