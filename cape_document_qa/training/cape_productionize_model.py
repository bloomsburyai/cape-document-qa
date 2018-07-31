from cape_document_qa import patches
from cape_document_qa.cape_config import VEC_DIR, TrainConfig, LM_VOCAB, LM_WEIGHTS, LM_OPTIONS, LM_TOKEN_WEIGHTS
from cape_document_qa.training.cape_convert_to_cpu import convert
from os.path import exists, join
from os import listdir, makedirs
from shutil import copy2
import tensorflow as tf
import argparse


def copy_checkpoint(model_dir, output_dir):
    best_dir = join(model_dir, 'best')
    if exists(best_dir) and len(listdir(best_dir)) > 0:
        checkpoint_dir = best_dir
    else:
        checkpoint_dir = join(model_dir, 'save')
    checkpoint_fi = tf.train.latest_checkpoint(checkpoint_dir)
    for fi in listdir(checkpoint_dir):
        if checkpoint_fi in fi:
            copy2(fi, join(output_dir, 'save'))


def copy_word_vectors(train_config, output_dir):
    word_vector_path = join(VEC_DIR, train_config.word_vectors)
    copy2(word_vector_path, output_dir)


def copy_elmo_resources(output_dir):
    copy2(LM_WEIGHTS, output_dir)
    copy2(LM_OPTIONS, output_dir)
    copy2(LM_TOKEN_WEIGHTS, output_dir) # possibly remove, not necessary


def productionize(model_dir, output_dir, train_config, convert_from_cudnn=TrainConfig):
    if exists(output_dir):
        raise Exception('Output destination already exists')
    if convert_from_cudnn:
        convert(model_dir, output_dir)
    else:
        # need to copy across stuff
        makedirs(join(output_dir, 'save'))
        copy2(join(model_dir, 'model.pkl'), join(output_dir, 'model.pkl'))
        copy_checkpoint(model_dir, output_dir)

    copy_word_vectors(train_config, output_dir)
    copy_elmo_resources(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, help='path to model directory to productionize')
    parser.add_argument("--output_dir", type=str, help='path to create production model at')
    parser.add_argument('--convert_from_cudnn', action='store_true', dest='cudnn',
                        help='Convert model from Cudnn (highly recommended)')
    parser.add_argument('--no_convert_from_cudnn', action='store_false', dest='cudnn', help='')
    parser.set_defaults(cudnn=True)
    args = parser.parse_args()
    productionize(args.target_model, args.output_dir, convert_from_cudnn=args.cudnn)







