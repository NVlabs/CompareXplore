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
from saver import saver
from config import FLAGS
from utils import OurTimer, format_seconds, print_stats


import torch.nn as nn

import torch
from openai import OpenAI
import openai
import numpy as np


class OpenAIEncoder(nn.Module):
    def __init__(self):
        super(OpenAIEncoder, self).__init__()

        self.dummy_parameter = torch.nn.Linear(1, 1)

        self.client = OpenAI()
        self.oepnai_call_in_secs_li = []
        self.openai_io_collector = []  # only relevant to **this** (the current) object
        self.cache = OpenAIIOCache(
            'embedding'
        )  # load previously saved io pairs (fancy!)

        if FLAGS.OpenAI_embedding_model == 'text-embedding-3-large':
            self.dim = 3072
        else:
            raise NotImplementedError()

        # embedding = get_embedding('hello\nworld', model='text-embedding-3-large')
        # print(len(embedding))
        # embedding2 = get_embedding('hello world', model='text-embedding-3-large')
        # print(embedding ==embedding2)
        # print(embedding[0:10])
        # print(embedding2[0:10])

    def forward(self, data):

        if self._is_double_list(data.text_to_use):
            txt_li = data.text_to_use[0]
            assert self._is_double_list(data.gname)
            gname_li = data.gname[0]
            assert self._is_double_list(data.gname)
        else:
            txt_li = data.text_to_use
            gname_li = data.gname

        # assert (
        #     type(data.text_to_use) is list and len(data.text_to_use) == 1
        # ), f'type(data.text_to_use)={type(data.text_to_use)}; len(data.text_to_use)={len(data.text_to_use)}; data.text_to_use={data.text_to_use}'

        num_chunk_li = torch.bincount(data.batch).tolist()
        assert len(num_chunk_li) == len(gname_li) == len(txt_li) > 0

        # saver.log_info(f'data.gname={data.gname}')
        # saver.log_info(f'len(data.text_to_use[0])={len(data.text_to_use[0])}')
        # saver.log_info(f'num_chunk_li={num_chunk_li}')

        embedding_li = []
        for gname, txt, num_chunk in zip(gname_li, txt_li, num_chunk_li):

            embedding = self.cache.get_from_cache(txt)
            if embedding is None:
                timer = OurTimer()
                embedding = self._get_embedding(txt)
                oepnai_call_in_secs = timer.time_and_clear(only_seconds=True)
                saver.log_info_at_most(
                    f'oepnai_call_in_secs={oepnai_call_in_secs}',
                    f'GPT response time once embed',
                    1,
                )
                self.oepnai_call_in_secs_li.append(oepnai_call_in_secs)
                self.openai_io_collector.append({'txt': txt, 'embedding': embedding})
                self.cache.insert_into_cache(txt, embedding)

                if (
                    len(self.openai_io_collector) % FLAGS.save_cache_every_new_io_pairs
                    == 0
                ):
                    self.save_cache()

            assert num_chunk > 0
            for _ in range(num_chunk):
                embedding_li.append(embedding)

            # saver.log_info(f'embedding.shape={embedding.shape}')
            assert embedding.shape == (self.dim,)

            # saver.log_info_at_most(
            #     f'txt for {gname}=\n{txt}\nembedding.shape=\n{embedding.shape}\nembedding[0:5]={embedding[0:5]}',
            #     f'GPT response embed {gname}',
            #     1,
            # )

        if len(self.oepnai_call_in_secs_li) % 100 == 0:
            avg_time = format_seconds(np.mean(self.oepnai_call_in_secs_li))
            saver.log_info_at_most(
                f'Average OpenAI call time={avg_time}',
                f'oepnai_call_in_secs_li embed',
                10,
            )

        return torch.tensor(
            np.array(embedding_li),
            dtype=torch.float,  # otherwise, "RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float" later
            device=data.x.device,
        )

        # exit()
        # pass

    def _is_double_list(self, li):
        return isinstance(li[0], list) if li else False

    def _get_embedding(self, text):
        #  text = text.replace("\n", " ")
        try:
            rtn = np.array(
                self.client.embeddings.create(
                    input=[text], model=FLAGS.OpenAI_embedding_model
                )
                .data[0]
                .embedding
            )
        except Exception as e:
            saver.log_info(f'Exception:\n{e}')
            saver.log_info(f'text:\n{text}')
            self.save_cache()
            raise e

        return rtn

    def save_cache(self):
        saver.save_dict_as_pickle(
            {'openai_io_collector': self.openai_io_collector}, f'openai_io_collector'
        )
        print_stats(self.oepnai_call_in_secs_li, 'oepnai_call_in_secs_li', saver=saver)
        saver.log_info(self.cache.report())
