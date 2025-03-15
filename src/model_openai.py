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

from utils_openai import OpenAIIOCache
from model import Model, get_y_with_target
from config import FLAGS
from saver import saver
from utils import OurTimer, format_seconds, print_stats

from openai import OpenAI

from tqdm import tqdm
import torch

from collections import OrderedDict
import numpy as np


class OpenAIModel(Model):  # Ranking/Comparison
    def __init__(self, init_pragma_dict=None, dataset=None, task=None):
        super(OpenAIModel, self).__init__()
        self.task = FLAGS.task
        self.target_list = self._get_target_list()

        self.dummy_parameter = torch.nn.Linear(1, 1)
        self.client = OpenAI()
        self.oepnai_call_in_secs_li = []
        self.openai_io_collector = []  # only relevant to **this** (the current) object
        self.cache = OpenAIIOCache(
            'comparison'
        )  # load previously saved io pairs (fancy!)

        # completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        #     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        # ]
        # )

        # print(completion.choices[0].message)

    def forward(
        self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None
    ):
        # total_loss = 0.0
        out_dict = OrderedDict()
        loss_dict = OrderedDict()

        mode = 'normal'
        self._decode(data, out_dict, loss_dict, mode)

        response_li = []
        if forward_pairwise and FLAGS.pairwise_class:
            assert type(data.txt) is list and len(data.txt) == 1
            txt_li = data.txt[0]
            assert (
                type(txt_li) is list and len(txt_li) % 2 == 0
            )  # first half: d1; second half: d2

            # saver.log_info(f'data.txt=\n{data.txt}\n{len(data)}\n{len(data[0])}')
            # exit()

            li_1, li_2 = self._split_li_into_2_halves(txt_li)

            for d_1, d_2 in zip(li_1, li_2):
                prompt = (
                    f'#design 1:\n{d_1}\n#design 2:\n{d_2}\nDoes design 1 have a higher latency that design 2?'
                    '[Must provide a **yes** or **no** answer. In other words, your response must include either '
                    'the keyword **yes** or the keyword **no**]'
                )

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert in High-Level Synthesis for FPGA, ' \
                    'skilled in analyzing the performance of FPGA designs in terms of latency.",
                    },
                    {"role": "user", "content": prompt},
                ]

                response = self.cache.get_from_cache(messages)
                if response is None:
                    timer = OurTimer()
                    completion = self.client.chat.completions.create(
                        model=FLAGS.OpenAI_model, messages=messages
                    )
                    oepnai_call_in_secs = timer.time_and_clear(only_seconds=True)
                    saver.log_info_at_most(
                        f'oepnai_call_in_secs={oepnai_call_in_secs}',
                        f'GPT response time once',
                        1,
                    )
                    self.oepnai_call_in_secs_li.append(oepnai_call_in_secs)

                    response = completion.choices[0].message.content

                    self.openai_io_collector.append(
                        {'messages': messages, 'response': response}
                    )

                response_li.append(response)

                # self.openai_io_collector.append(
                #     {'messages': messages, 'response': response}
                # )

                saver.log_info_at_most(
                    f'prompt=\n{prompt}\nresponse=\n{response}', f'GPT response', 2
                )

            if len(self.oepnai_call_in_secs_li) % 100 == 0:
                avg_time = format_seconds(np.mean(self.oepnai_call_in_secs_li))
                saver.log_info_at_most(
                    f'Average OpenAI call time={avg_time}',
                    f'oepnai_call_in_secs_li',
                    10,
                )

                # exit()
            mode = 'pairwise_class'
            self._decode(data, out_dict, loss_dict, mode, response_li)

        return out_dict, torch.tensor(0.0), loss_dict, torch.tensor(0.0)

    def _decode(self, data, out_dict, loss_dict, mode, response_li=None):
        for target_name in self.target_list:
            out = torch.zeros_like(get_y_with_target(data, 'perf'))
            if mode == 'normal':
                target_name_s = target_name
                out_dict[target_name_s] = out

            elif mode == 'pairwise_class':
                target_name_s = f'{target_name}_pairwise_class'
                assert type(response_li) is list and len(response_li) > 0

                answer_li = (
                    []
                )  # tricky! for every target, we get the same reaponse (TODO: set target_list dynamically somehow to be efficient)
                for response in response_li:
                    if 'yes' in response.lower():
                        answer = [0, 1]
                    elif 'no' in response.lower():
                        answer = [1, 0]
                    else:
                        answer = [0, 1]
                    answer_li.append(answer)

                answer_mat = torch.tensor(answer_li)
                out_dict[target_name_s] = answer_mat
                # saver.log_info(f'answer_mat={answer_mat}')
                # exit()

            else:
                assert False

            loss_dict[target_name_s] = torch.tensor(0.0)

    def _split_li_into_2_halves(self, li):
        length = len(li)
        assert length % 2 == 0  # divisible by 2 -- otherwise data loader has some issue
        half_point = int(length / 2)
        d1 = li[0:half_point]
        d2 = li[half_point:]
        assert len(d1) == len(d2)
        return d1, d2

    # Deleting (Calling destructor)
    def __del__(self):
        saver.save_dict_as_pickle(
            {'openai_io_collector': self.openai_io_collector}, f'openai_io_collector'
        )
        print_stats(self.oepnai_call_in_secs_li, 'oepnai_call_in_secs_li', saver=saver)
        saver.log_info(self.cache.report())
