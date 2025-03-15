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

'''

05/29/2023: A database object to use either the redis database or a custom object.

'''

from config import FLAGS
from saver import saver
from collections import OrderedDict

class OurDatabase(object):
    def __int__(self):
        self.flushdb()

    def flushdb(self):
        self.d = {}
        self.d_decoded = {}

    def hmset(self, n, data):
        self.d[n] = data
        new_data = OrderedDict()
        assert type(data) is dict
        for k, v in data.items():
            new_data[k.decode('utf-8')] = v
        self.d_decoded[n] = new_data

    def hkeys(self, n):
        if n not in self.d:
            return []
        return list(self.d[n].keys())

    def hget(self, n, key):
        try:
            return self.d_decoded[n][key]
        except KeyError as e:
            exit()


def create_database():
    if FLAGS.use_redis:
        import redis
        rtn = redis.StrictRedis(host='localhost', port=6379)
        saver.log_info(f'Connection established to port 6379')
        return rtn
    else:
        return OurDatabase()
