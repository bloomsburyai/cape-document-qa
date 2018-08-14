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

from cape_document_qa import patches
from cape_document_qa.preprocessing.cape_preprocess_squad import preprocess_squad_dataset
from cape_document_qa.cape_docqa_machine_reader import CapeDocQAMachineReaderConfig, get_production_model_config
from os.path import join
from docqa.data_processing.document_splitter import MergeParagraphs
from cape_document_qa.evaluation.cape_docqa_eval import get_multidataset, compute_and_dump_official_output,\
    get_aggregated_df, get_questions, get_para_filter, RecordParagraphSpanPrediction
from typing import Union, List
from docqa.utils import ResourceLoader
import pickle
from docqa.evaluator import AysncEvaluatorRunner
from docqa.data_processing.word_vectors import load_word_vectors
import tensorflow as tf
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec
from cape_document_qa.cape_config import PREPRO_DATASET_DIR
import pandas as pd


def build_model_and_evaluator_runner(model_config, max_answer_len, n_paragraphs):
    with open(model_config.model_pickle_file, 'rb') as f:
        model = pickle.load(f)

    model.lm_model.weight_file = model_config.lm_weights_file
    model.lm_model.lm_vocab_file = model_config.vocab_file
    model.lm_model.embed_weights_file = model_config.lm_token_weights_file
    model.lm_model.options_file = model_config.lm_options_file
    model.word_embed.vec_name = model_config.word_vector_file
    vocab_to_ignore = {'<S>', '</S>', '<UNK>', '!!!MAXTERMID'}

    vocab_to_init_with = {
        line.strip() for line in open(model_config.vocab_file, encoding="utf-8")
        if line.strip() not in vocab_to_ignore
    }

    #evaluator_runner = AysncEvaluatorRunner([RecordParagraphSpanPrediction(max_answer_len, True)], model, 10)
    sess = tf.Session()
    with sess.as_default():
        model.set_input_spec(
            ParagraphAndQuestionSpec(None, None, None, 14), vocab_to_init_with,
            word_vec_loader=ResourceLoader(load_vec_fn=lambda x, y: load_word_vectors(x, y, is_path=True))
        )
        evaluator_runner = AysncEvaluatorRunner([RecordParagraphSpanPrediction(max_answer_len, True)], model, 10)

        input_dict = {p: x for p, x in zip(model.get_placeholders(), evaluator_runner.dequeue_op)}
        pred = model.get_predictions_for(input_dict)
    evaluator_runner.set_input(pred)

    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars_to_restore = [x for x in all_vars if x.name not in lm_var_names]
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, model_config.checkpoint_file)
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))

    return sess, model, evaluator_runner


def perform_evaluation(model_config: CapeDocQAMachineReaderConfig,
                       dataset_name: str,
                       tokens_per_paragraph: int,
                       filter_type: str,
                       n_processes: int,
                       n_paragraphs: int,
                       batch_size: int,
                       max_answer_len: int,
                       official_output_path: str,
                       paragraph_output_path: str,
                       aggregated_output_path: str,
                       n_samples: Union[int, None],
                       per_document: False,
                       ):
    """Perform an evaluation using cape's answer decoder

    A file will be created listing the answers per question ID for each dataset

    :param model_name: path to the model to evaluate
    :param dataset_names: list of strings of datasets to evaluate
    :param tokens_per_paragraph: how big to make paragraph chunks
    :param filter_type: how to select the paragraphs to read
    :param n_processes: how many processes to use when multiprocessing
    :param n_paragraphs: how many paragraphs to read per question
    :param batch_size: how many datapoints to evaluate at once
    :param checkpoint: string, checkpoint to load
    :param no_ema: if true, dont use EMA weights
    :param max_answer_len: the maximum allowable length of an answer in tokens
    :param official_output_path: path to write official output to
    :param paragraph_output_path: path to write paragraph output to
    :param aggregated_output_path: path to write aggregated output to
    :param elmo_char_cnn: if true, uses the elmo CNN to make token embeddings, less OOV but
        requires much more memory
    :param per_document: if true, scores each document associated with a question seperately,
        if false, just the highest scoring answer from any document is used.
    """

    print('Setting Up:')
    dataset = get_multidataset([dataset_name])
    splitter = MergeParagraphs(tokens_per_paragraph)
    para_filter = get_para_filter(filter_type, per_document, n_paragraphs)

    sess, model, evaluator_runner = build_model_and_evaluator_runner(model_config, max_answer_len, n_paragraphs)
    test_questions, n_questions = get_questions(
        per_document, dataset, splitter,
        para_filter, model.preprocessor,
        n_processes, batch_size
    )
    print('Starting Eval')
    evaluation = evaluator_runner.run_evaluators(sess, test_questions, dataset_name, n_samples, {})

    print('Exporting and Post-processing')
    if not all(len(x) == n_questions for x in evaluation.per_sample.values()):
        raise RuntimeError()

    df = pd.DataFrame(evaluation.per_sample)
    compute_and_dump_official_output(df, official_output_path)

    print("Saving paragraph result")
    df.to_csv(paragraph_output_path, index=False)

    print("Computing scores")
    agg_df = get_aggregated_df(df, per_document)
    agg_df.to_csv(aggregated_output_path, index=False)
    sess.close()
    del sess
    tf.reset_default_graph()


def perform_benchmark_evaluation(dataset_name: str,
                                 test_files: List[str],
                                 model_path: Union[str, None]=None,
                                 official_output_path: Union[str, None]=None,
                                 paragraph_output_path: Union[str, None]=None,
                                 aggregated_output_path: Union[str, None]=None,
                                 filter_type: Union[str, None]=None,
                                 n_processes: int=8,
                                 tokens_per_paragraph: int=400,
                                 n_paragraphs: int=15,
                                 batch_size: int=120,
                                 max_answer_len: int=50,
                                 n_samples: Union[int, None]=None,
                                 per_document: bool=False,
                                 ):
    """"""
    official_output_path = '{}_official_output.json'.format(dataset_name) \
        if official_output_path is None else official_output_path
    aggregated_output_path = '{}_aggregated_output.csv'.format(dataset_name) \
        if aggregated_output_path is None else aggregated_output_path
    paragraph_output_path = '{}_paragraph_output.csv'.format(dataset_name) \
        if paragraph_output_path is None else paragraph_output_path

    fold_dict = {'train': [], 'dev': test_files, 'test': []}
    preprocess_squad_dataset(dataset_name, fold_dict)

    vocab_file = join(PREPRO_DATASET_DIR, dataset_name, 'vocab.txt')

    if model_path is None:
        model_config = get_production_model_config()
        model_config.vocab_file = vocab_file
    else:
        model_config = CapeDocQAMachineReaderConfig(
            model_pickle_file=join(model_path, 'model.pkl'),
            model_checkpoint_file=join(model_path, 'save', 'checkpoint-123456789'),
            lm_weights_file=join(model_path, 'elmo_weights.hdf5'),
            lm_token_weights_file=None,
            lm_options_file=join(model_path, 'elmo_options.json'),
            vocab_file=vocab_file,
            word_vector_file=join(model_path, 'glove.840B.300d')
        )

    perform_evaluation(
        model_config=model_config,
        dataset_name=dataset_name,
        tokens_per_paragraph=tokens_per_paragraph,
        filter_type=filter_type,
        n_processes=n_processes,
        n_paragraphs=n_paragraphs,
        batch_size=batch_size,
        max_answer_len=max_answer_len,
        official_output_path=official_output_path,
        paragraph_output_path=paragraph_output_path,
        aggregated_output_path=aggregated_output_path,
        n_samples=n_samples,
        per_document=per_document
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Perform an evaluation on squad-formatted data using a production-ready model")
    parser.add_argument('-n', '--dataset_name', type=str, dest='dataset_name', help='name of dataset to preprocess')
    parser.add_argument('-tf', '--test_files', nargs='+', help='Which datasets to compute')
    parser.add_argument('-m', '--model', type=str, default=None, dest='model', help='name of dataset to preprocess')

    parser.add_argument('-p', '--paragraph_output', type=str, default=None,
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument('-ag', '--aggregated_output', type=str, default=None,
                        help="Save aggregated results in csv format")
    parser.add_argument('-o', '--official_output', type=str, default=None, help="Build an offical output file with the model's"
                                                                  " most confident span for each (question, doc) pair")
    parser.add_argument('--n_processes', type=int, default=8,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-T', '--tokens', type=int, default=400,
                        help="Max tokens per a paragraph")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=15,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-f', '--filter', type=str, default=None, choices=["tfidf", "truncate", "linear"],
                        help="How to select paragraphs")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help="Batch size, larger sizes might be faster but wll take more memory")
    parser.add_argument('--max_answer_len', type=int, default=50,
                        help="Max answer span to select")
    parser.add_argument('--per_document', action='store_true', dest='per_document', help='Score each question document pair')
    parser.add_argument('--no_per_document', action='store_false', dest='per_document', help='each question'
                                                                                             'is answered by the highest'
                                                                                             'scoreing answer from all associated docs')
    parser.add_argument('-s', '--samples', type=int, default=None, help='Number of samples to run, defaults to all')
    parser.set_defaults(per_document=True)
    args = parser.parse_args()

    perform_benchmark_evaluation(
        args.dataset_name,
        args.test_files,
        model_path=args.model,
        official_output_path=args.official_output,
        paragraph_output_path=args.paragraph_output,
        aggregated_output_path=args.aggregated_output,
        filter_type=args.filter,
        n_processes=args.n_processes,
        tokens_per_paragraph=args.tokens,
        n_paragraphs=args.n_paragraphs,
        batch_size=args.batch_size,
        max_answer_len=args.max_answer_len,
        n_samples=args.samples,
        per_document=args.per_document
    )

