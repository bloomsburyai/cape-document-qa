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
