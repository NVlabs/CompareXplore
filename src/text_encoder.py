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

from config import FLAGS
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm import tqdm
from saver import saver

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class TextEncoder(object):
    def encode(self, fulltext):
        raise NotImplementedError()

    def dim(self):
        raise NotImplementedError()


class RobertaTextEncoder(TextEncoder):
    def __init__(self):
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # def encode(self, list_of_text):
    #     X = []
    #     for fulltext in list_of_text:
    #         inputs = self.roberta_tokenizer(fulltext, return_tensors="pt")
    #         output_emb = self.roberta_model(**inputs)['pooler_output'][:,:4]
    #         X.append(output_emb)
    #     X = torch.cat(X, dim=0)
    #     return X

    def encode(self, fulltext):
        inputs = self.roberta_tokenizer(fulltext, return_tensors="pt")
        output_emb = self.roberta_model(**inputs)['pooler_output'] #[:, :4]
        return output_emb

    def dim(self):
        return 768


class Word2VecTextEncoder(TextEncoder):
    def __init__(self):
        self.sentences = []
        self.model = None
        self.dim_ = FLAGS.fulltext_dim
        assert self.dim_ > 0


    def add_sentence(self, sentence):
        assert type(sentence) is str
        self.sentences.append(TaggedDocument(sentence.split(), [len(self.sentences)]))


    def encode(self, fulltext):
        if self.model is None: # has not been trained
            saver.log_info('Training Doc2Vec...')
            self.model = Doc2Vec(self.sentences, vector_size=self.dim_, window=2, min_count=1, workers=4)
            saver.log_info(f'Doc2Vec trained with dv {len(self.model.dv)} and wv {len(self.model.wv)}')
            saver.log_info(f'Example sentence: {self.sentences[0]}')

        vector = self.model.infer_vector(fulltext.split())
        return torch.tensor(vector.reshape(1, len(vector)))

    def dim(self):
        return self.dim_


def create_text_encoder(t):
    if t == 'roberta':
        return RobertaTextEncoder()
    elif t == 'word2vec':
        return Word2VecTextEncoder()
    else:
        return None


