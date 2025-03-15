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


from pytriton.client import FuturesModelClient
import numpy as np
import json


def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


print(f'valid_jsonl loading...')

valid_jsonl = read_jsonl(
    '/home/yba/software-gnn/file/pairwise_data/02292024_robert/validation.comparisons.rm.jsonl'
)

print(f'valid_jsonl loaded; len(valid_jsonl)={len(valid_jsonl)}')
chosen = valid_jsonl[0::2][:10]
rejected = valid_jsonl[1::2][:10]

host = 'localhost'
port = '1234'
model_name = 'reward_model'


def to_array(str_list: list[str]) -> np.ndarray:
    array = np.array(str_list)
    array = np.char.encode(array, "utf-8")
    array = np.expand_dims(array, -1)
    return array


correct_count = 0

iter = 0
for c, r in zip(chosen, rejected):
    print(f'iter={iter}')

    with FuturesModelClient(f"{host}:{port}", model_name, max_workers=10) as client:

        # print(f"c['text']={c['text']}")
        # print(f"r['text']={r['text']}")

        c_to_use = c['text']
        r_to_use = r['text']

        new_txt = "I'm giving you an HLS design for matrix multiplication, can you give me the reward?"
        c_to_use = f'{new_txt}\n{c_to_use}'
        r_to_use = f'{new_txt}\n{r_to_use}'

        sentences = to_array([c_to_use, r_to_use])
        print(f"sentences={sentences}")
        rewards = client.infer_batch(sentences=sentences).result()['rewards']
        print(f"rewards={rewards}")

        correct_count += rewards[0] > rewards[1]

    iter += 1
print(correct_count / len(chosen))
