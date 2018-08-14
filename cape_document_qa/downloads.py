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
import shutil
import subprocess
from cape_document_qa.cape_config import VEC_DIR, LM_DIR, SQUAD_SOURCE_DIR, TrainConfig, \
    ROOT_DATA_DIR, TRIVIA_QA, CAPE_DATA_RELEASE, CAPE_DATA_SERVER, GLOVE_SERVER, SQUAD_SERVER, TRIVIAQA_SERVER, \
    PRODUCTION_MODEL_NAME
import argparse


def _wget(url, out='.'):
    subprocess.check_call(['wget', '-P', out, url])


def download_resource(filename, server, destination):
    destination_path = os.path.join(destination, filename)
    if not os.path.exists(destination_path):
        os.makedirs(destination, exist_ok=True)
        url = server + filename
        _wget(url, out=destination)
    return destination_path


def download_glove(glove_file, destination):
    print('Downloading Glove {}'.format(glove_file))
    if not os.path.exists(os.path.join(destination, glove_file + '.txt')):
        glove_zip = glove_file + '.zip'
        destination_path = download_resource(glove_zip, GLOVE_SERVER, destination)
        subprocess.check_call(['unzip', destination_path, '-d', destination])
        os.remove(destination_path)


def download_squad(destination):
    print('Downloading SQUAD:')
    for fold in ['train', 'dev']:
        filename = '{}-v1.1.json'.format(fold)
        download_resource(filename, SQUAD_SERVER, destination)


def download_triviaqa(destination):
    print('Downloading TriviaQA')
    filename = 'triviaqa-rc.tar.gz'
    destination_path = download_resource(filename, TRIVIAQA_SERVER, destination)
    os.makedirs(TRIVIA_QA, exist_ok=True)
    subprocess.check_call(['tar', '-xf', destination_path, '--directory', TRIVIA_QA])
    os.remove(destination_path)


def _get_cape_server():
    return CAPE_DATA_SERVER + CAPE_DATA_RELEASE + '/'


def download_lm(destination):
    print('Downloading Elmo language model')
    filename = 'lm.tar.bz2'
    destination_path = download_resource(filename, _get_cape_server(), destination)
    subprocess.check_call(['tar', '-jxvf', destination_path, '--directory', destination])
    os.remove(destination_path)


def download_model(model_name, destination, word_vectors=None, compression='.tar.bz2'):
    print('Downloading model')
    destination_path = download_resource(model_name + compression, _get_cape_server(), destination)
    subprocess.check_call(['tar', '-jxvf', destination_path, '--directory', destination])
    os.remove(destination_path)
    if word_vectors:
        model_dir = os.path.join(destination, model_name)
        download_glove(word_vectors, model_dir)


def downloads_for_training():
    """Downloads glove files, squad, triviaqa and language models, and places them into appropriate locations"""
    print('Downloading data and resources for Training cape document qa models')
    glove_file = TrainConfig().word_vectors
    download_glove(glove_file, VEC_DIR)
    download_squad(SQUAD_SOURCE_DIR)
    download_triviaqa(ROOT_DATA_DIR)
    download_lm(LM_DIR)


def downloads_for_production(destination='.'):
    """Downloads glove file and a production model from the server, unzips and combines them"""
    glove_file = TrainConfig().word_vectors
    download_model(PRODUCTION_MODEL_NAME, destination, word_vectors=glove_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download resources to support cape document qa")
    parser.add_argument(default='production', dest='production_or_training', choices=["training", "production"],
                        help='What to download. "production" will download a pre-trained model '\
                             'ready to be used by cape.  "training" will download the resources '\
                             ' and datasets required to train your own models')
    parser.add_argument('-d', '--destination', dest='destination', default='.',
                        help='destination to download model to (only applicable to model download')
    args = parser.parse_args()

    if args.production_or_training == 'production':
        downloads_for_production(args.destination)
    elif args.production_or_training == 'training':
        downloads_for_training()
    else:
        raise Exception('Must be either production or training, not {}'.format(args.production_or_training))





