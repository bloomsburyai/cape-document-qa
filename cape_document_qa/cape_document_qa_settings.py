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

import os

def envint(varname: str, default: int) -> int:
   return int(os.getenv(varname, default))


THIS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))

MODELS_FOLDER = os.getenv('CAPE_MODELS_FOLDER', os.path.join(THIS_FOLDER, 'storage', 'models'))
MODEL_FOLDER = os.getenv('CAPE_MODEL_FOLDER', os.path.join(MODELS_FOLDER, 'production_ready_model'))
MODEL_URL = os.getenv('CAPE_MODEL_URL',
                      'https://github.com/bloomsburyai/cape-document-qa/releases/download/v0.1.2/production_ready_model.tar.xz')
MODEL_MB_SIZE = envint('CAPE_MODEL_MB_SIZE', 422)
DOWNLOAD_ALL_GLOVE_EMBEDDINGS = os.getenv('DOWNLOAD_ALL_GLOVE_EMBEDDINGS', 'False').lower() == 'true'
GLOVE_EMBEDDINGS_URL = os.getenv('CAPE_GLOVE_EMBEDDINGS_URL', 'https://nlp.stanford.edu/data/glove.840B.300d.zip')

LM_URL = os.getenv('CAPE_LM_URL', 'https://github.com/bloogram/cape-document-qa/releases/download/v0.1.2/lm.tar.bz2')
