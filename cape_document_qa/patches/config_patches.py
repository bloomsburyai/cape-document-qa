# Copyright 2018 BLEMUNDSBURY AI LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from docqa import config
from cape_document_qa import cape_config

config.CORPUS_DIR = cape_config.CORPUS_DIR
config.VEC_DIR = cape_config.VEC_DIR
config.SQUAD_SOURCE_DIR = cape_config.SQUAD_SOURCE_DIR
config.SQUAD_TRAIN = cape_config.SQUAD_TRAIN
config.SQUAD_DEV = cape_config.SQUAD_DEV
config.TRIVIA_QA = cape_config.TRIVIA_QA
config.TRIVIA_QA_UNFILTERED = cape_config.TRIVIA_QA_UNFILTERED
config.LM_DIR = cape_config.LM_DIR
config.LM_OPTIONS = cape_config.LM_OPTIONS
config.LM_WEIGHTS = cape_config.LM_WEIGHTS
config.LM_VOCAB = cape_config.LM_VOCAB
