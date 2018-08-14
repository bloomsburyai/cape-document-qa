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
from tqdm import tqdm
from typing import Union
import pickle
import os
import tensorflow as tf
import json
import argparse
from docqa.triviaqa.build_span_corpus import TriviaQaSpanCorpus
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, ParagraphAndQuestion
from cape_document_qa.cape_config import LM_WEIGHTS, LM_OPTIONS, LM_VOCAB, LM_TOKEN_WEIGHTS
from cape_machine_reader.cape_answer_decoder import softmax, find_answer_spans


def _get_checkpoing_filepath(model_dir):
    return tf.train.latest_checkpoint(os.path.join(model_dir, 'save'))


def _get_pickle_filepath(model_dir):
    return os.path.join(model_dir + '/model.pkl')


def init_model(sess, model_dir):
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars_to_restore = [x for x in all_vars if x.name not in lm_var_names]
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, _get_checkpoing_filepath(model_dir))
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))


def prepro(ds, fold, num_tokens_per_group, num_paragraphs, pad=True, n_samples=None):
    fold_funcs = {
        'train': lambda: ds.get_train(), 'dev': lambda: ds.get_dev(), 'test': lambda: ds.get_test()
    }
    qs = fold_funcs[fold]()
    if n_samples is not None:
        qs = qs[:n_samples]
    evidence = ds.evidence

    prep = None
    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(num_tokens_per_group),
                                                ShallowOpenWebRanker(num_paragraphs),
                                                prep, intern=True)

    answers = {}
    batches = {}
    for q in tqdm(qs, ncols=80, desc='preprocessing'):
        pre = extract.preprocess([q], evidence)
        if len(pre.data) == 0:
            continue
        assert len(pre.data) < 2
        assert q.question_id not in answers
        assert q.question_id not in batches
        mpq = pre.data[0]
        pq_batch = [
            ParagraphAndQuestion(p.get_context(), q.question, None, q.question_id, p.doc_id) for p in mpq.paragraphs
            # document paragraph question?
        ]
        if pad:
            for i in range(num_paragraphs - len(pq_batch)):
                pq_batch.append(ParagraphAndQuestion([], q.question, None, q.question_id, None))

        answers[q.question_id] = mpq.answer_text
        batches[q.question_id] = pq_batch

    voc = {w for bs in batches.values() for b in bs for w in b.question}
    voc.update({w for bs in batches.values() for b in bs for w in b.context})
    return answers, batches, voc


def _decode_answers(qid, batch, yp, yp2, answers, topk, verbose=False):

    def _flatten_logits(logits, lens):
        flat = []
        for log, le in zip(list(logits), lens):
            flat += list(log[:le])
        return flat

    mr_question_answers = []
    context = [w for b in batch for w in b.context]
    batch_lens = [len(b.context) for b in batch]
    flat_yp, flat_yp2 = _flatten_logits(yp, batch_lens), _flatten_logits(yp2, batch_lens)

    for k, (answer_span, answer_score) in enumerate(find_answer_spans(softmax(flat_yp), softmax(flat_yp2))):
        mr_answer_at_k = ' '.join(context[answer_span[0]: answer_span[1] + 1])
        mr_question_answers.append(mr_answer_at_k)
        if k > topk:
            break
    if verbose:
        print('_' * 80)
        print('Question: ' + ' '.join(batch[0].question))
        print('Given Answers:')
        print('"' + '","'.join(answers[qid]) + '"')
        print('Machine Read Answers:')
        print('\n'.join(['{}: {}'.format(i + 1, mra) for i, mra in enumerate(mr_question_answers)]))
    return mr_question_answers


def get_answers(sess, model, pred, batches,  answers, topk, verbose=False):
    mr_answers = {}
    qids = batches.keys() if verbose else tqdm(batches.keys(), ncols=80, desc='Extracting answers')
    for qid in qids:
        batch = batches[qid]
        feed = model.encode(batch, False)
        yp, yp2 = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)
        mr_answers[qid] = _decode_answers(qid, batch, yp, yp2, answers, topk, verbose)
    return mr_answers


def get_answers_faster(sess, model, pred, batches, answers, topk, batch_size, verbose=False):

    mr_answers = {}
    batch_acc = []
    location_acc = []
    logits_dict = {}

    def _run_batch(batch_acc, location_acc):
        feed = model.encode(batch_acc, False)
        yp, yp2 = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)
        qqids, js = zip(*location_acc)
        for (qqid, j, stl, enl) in zip(qqids, js, yp[:len(location_acc)], yp2[:len(location_acc)]):
            logits_dict[(qqid, j)] = (stl, enl)

    for qid, batch in tqdm(batches.items()):
        for i, b in enumerate(batch):
            batch_acc.append(b)
            location_acc.append((qid, i))
            if len(batch_acc) == batch_size:
                _run_batch(batch_acc, location_acc)
                batch_acc, location_acc = [], []

    if len(batch_acc) != 0:
        bs = len(batch_acc)
        for _ in range(batch_size - bs):
            batch_acc.append(ParagraphAndQuestion([], [], None, None, None))
        _run_batch(batch_acc, location_acc)

    qids = batches.keys() if verbose else tqdm(batches.keys(), ncols=80, desc='Extracting answers')
    for qid in qids:
        batch = batches[qid]
        yp = [logits_dict[qid, j][0] for j in range(len(batch))]
        yp2 = [logits_dict[qid, j][1] for j in range(len(batch))]
        mr_answers[qid] = _decode_answers(qid, batch, yp, yp2, answers, topk, verbose)
    return mr_answers


def load_model(model_dir, elmo_char_cnn):
    with open(_get_pickle_filepath(model_dir), 'rb') as f:
        model = pickle.load(f)

    model.lm_model.weight_file = LM_WEIGHTS
    model.lm_model.lm_vocab_file = LM_VOCAB
    model.lm_model.embed_weights_file = None if elmo_char_cnn else LM_TOKEN_WEIGHTS
    model.lm_model.options_file = LM_OPTIONS
    return model


def extract_answers(num_paragraphs: int,
                    num_tokens_per_group: int,
                    topk: int,
                    dataset_name: str,
                    dataset_fold: str,
                    model_dir: str,
                    savepath: str,
                    verbose: bool,
                    batch_opt: bool,
                    batch_size: int,
                    elmo_character_cnn: bool,
                    n_samples: Union[None, int]
                    ):
    """Extract answers for a Cape-flavoured DocumentQA model. Uses Cape's answering mechanism
    to generate the top k scoring answering spans for document,question pairs from preprocessed Training/Dev/Test folds.
    Writes results to file in a json format: the id for the question followed by a list of k answer spans, highest scoring first.

    :param num_paragraphs: Maximum number of paragraphs to sample from a document
    :param num_tokens_per_group: Maximum size of a paragraph in tokens
    :param topk: Number of answers to extract from a document for a given question
    :param dataset_name: the name of the dataset under evaluation, e.g. 'wiki', 'web', 'squad'
    :param dataset_fold: either 'train', 'dev' or 'test'
    :param model_dir: the path to the model directory of the model under test
    :param savepath: the path to save the results json to
    :param verbose: print answers to std out (useful for debugging)
    :param batch_opt: an optimiztion to compute more than one question answer per batch - faster but danger of OOM on smaller cards
    :param batch_size: The number of paragraphs per batch if the batch_opt option is True
    :param elmo_character_cnn: If True, uses the elmo character CNN to compute elmo token representations on the fly. If false,
        it uses the precomputed token representations from training. Uses up quite a lot of GPU memory if True, so you may need to adjust
        batch size
    """
    print('Prepro:')
    batch_size = batch_size if batch_opt else num_paragraphs
    ds = TriviaQaSpanCorpus(dataset_name)

    gold_answer_dict, batch_dict, voc = prepro(
        ds, dataset_fold, num_tokens_per_group, num_paragraphs, pad=not batch_opt, n_samples=n_samples)

    print("Loading Model")
    model = load_model(model_dir, elmo_character_cnn)

    print('Building Graph')
    sess = tf.Session()
    with sess.as_default():
        model.set_input_spec(ParagraphAndQuestionSpec(batch_size, None, None, 14), voc)
        pred = model.get_prediction()

    print("Restoring Weights:")
    init_model(sess, model_dir)

    if batch_opt:
        mr_answer_dict = get_answers_faster(sess, model, pred, batch_dict, gold_answer_dict, topk, batch_size, verbose=verbose)
    else:
        mr_answer_dict = get_answers(sess, model, pred, batch_dict, gold_answer_dict, topk, verbose=verbose)

    with open(savepath, 'w', encoding='utf8') as f:
        json.dump(mr_answer_dict, f)

    sess.close()
    del sess
    tf.reset_default_graph()


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation using Cape's model decoder")
    parser.add_argument('model', help='path to model directory')
    parser.add_argument('-k', default=5, type=int, help='Number answers to compute per question')
    parser.add_argument('-p', '--num_paragraphs', default=16, type=int, help='Number of paragraphs to sample per question')
    parser.add_argument('-t', '--tokens_per_paragraph', default=500, type=int, help='Number of tokens per paragraph')
    parser.add_argument('-d', '--datasets', nargs='+', default=['squad'], help='Which datasets to compute')
    parser.add_argument('-bo', '--batch_opt', type=bool, default=True, help='Compute more than one question at a time (faster, but can get OOM)')
    parser.add_argument('-bs', '--batch_size', default=16, type=int, help='batch size to use for batch_opt option')
    parser.add_argument('-f', '--fold', default='dev', dest='fold', choices=["train", "dev", "test"], help='which fold to evaluate on')
    parser.add_argument('-c', '--elmo_character_cnn',  action='store_true', dest='elmo_character_cnn', help='Use Elmo char CNN - if false, uses precomputed token representations')
    parser.add_argument('-no_c', '--no_elmo_character_cnn', action='store_false', dest='elmo_character_cnn')
    parser.add_argument('-s', '--samples', type=int, default=None, help='Number of samples to run, defaults to all')
    parser.set_defaults(elmo_character_cnn=True)
    args = parser.parse_args()

    savepath = '{}_{}_top_{}_answers.json'
    verbose = False
    for dataset_name in args.datasets:
        extract_answers(
            args.num_paragraphs,
            args.tokens_per_paragraph,
            args.k,
            dataset_name,
            args.fold,
            args.model,
            savepath.format(args.model, dataset_name, args.k),
            verbose,
            args.batch_opt,
            args.batch_size,
            args.elmo_character_cnn,
            args.samples
        )


if __name__ == "__main__":
    main()
