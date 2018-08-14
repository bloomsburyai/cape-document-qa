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

import mock
import builtins
from tensorflow.python.keras import initializers


orig_import = __import__


def import_mock(name, *args):
    if name == 'tensorflow.contrib.keras.python.keras.initializers':
        return initializers
    return orig_import(name, *args)


with mock.patch('builtins.__import__', side_effect=import_mock):
    from docqa.nn.recurrent_layers import CudnnGru, BiRecurrentMapper, CompatGruCellSpec
