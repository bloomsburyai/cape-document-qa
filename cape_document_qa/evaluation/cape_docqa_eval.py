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
import argparse
import json
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from docqa.evaluator import AysncEvaluatorRunner, EvaluatorRunner
from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, FirstN
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import FixedOrderBatcher
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.evaluator import Evaluator, Evaluation
from docqa.model_dir import ModelDir
from docqa.triviaqa.build_span_corpus import TriviaQaSpanCorpus
from docqa.triviaqa.training_data import DocumentParagraphQuestion, ExtractMultiParagraphs, \
    ExtractMultiParagraphsPerQuestion
from docqa.triviaqa.trivia_qa_eval import exact_match_score as trivia_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as trivia_f1_score
from docqa.utils import ResourceLoader
from cape_document_qa.multi_dataset import MultiDataset


"""
Evaluate on TriviaQA data
"""


class RecordParagraphSpanPrediction(Evaluator):

    def __init__(self, bound: int, record_text_ans: bool):
        self.bound = bound
        self.record_text_ans = record_text_ans

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        needed = dict(spans=span, model_scores=score)
        return needed

    def evaluate(self, data: List[DocumentParagraphQuestion], true_len, **kargs):
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []

        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring"):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            text = point.get_context()
            pred_span = spans[i]
            pred_text = " ".join(text[pred_span[0]:pred_span[1] + 1])
            if self.record_text_ans:
                text_answers.append(pred_text)
                if point.answer is None:
                    continue

            f1 = 0
            em = False
            for answer in data[i].answer.answer_text:
                f1 = max(f1, trivia_f1_score(pred_text, answer))
                if not em:
                    em = trivia_em_score(pred_text, answer)

            pred_f1s[i] = f1
            pred_em[i] = em

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        if self.record_text_ans:
            results["text_answer"] = text_answers
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["text_f1"] = pred_f1s
        results["rank"] = [x.rank for x in data]
        results["text_em"] = pred_em
        results["para_start"] = [x.para_range[0] for x in data]
        results["para_end"] = [x.para_range[1] for x in data]
        results["question_id"] = [x.question_id for x in data]
        results["doc_id"] = [x.doc_id for x in data]
        return Evaluation({}, results)


def get_para_filter(filter_name, per_document, n_paragraphs):
    filter_name = ('tfidf' if per_document else 'linear') if filter_name is None else filter_name
    if filter_name == "tfidf":
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), n_paragraphs)
    elif filter_name == "truncate":
        para_filter = FirstN(n_paragraphs)
    elif filter_name == "linear":
        para_filter = ShallowOpenWebRanker(n_paragraphs)
    else:
        raise ValueError()
    return para_filter


def get_multidataset(dataset_names):
    datas = []
    for name in dataset_names:
        ds = TriviaQaSpanCorpus(name)
        ds.corpus_name = ds.corpus_name
        datas.append(ds)

    return MultiDataset(datas)


def get_checkpoint(step, model_dir):
    if step is not None:
        if step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()
    return checkpoint


def get_questions(per_document, dataset, splitter, para_filter, preprocessor, n_processes, batch_size):
    test_questions = dataset.get_dev()
    corpus = dataset.evidence
    print("Building question/paragraph pairs...")

    # Loads the relevant questions/documents, selects the right paragraphs, and runs the model's preprocessor
    if per_document:
        prep = ExtractMultiParagraphs(splitter, para_filter, preprocessor, require_an_answer=False)
    else:
        prep = ExtractMultiParagraphsPerQuestion(splitter, para_filter, preprocessor, require_an_answer=False)
    prepped_data = preprocess_par(test_questions, corpus, prep, n_processes, 1000)

    data = []
    for q in prepped_data.data:
        for i, p in enumerate(q.paragraphs):
            if q.answer_text is None:
                ans = None
            else:
                ans = TokenSpans(q.answer_text, p.answer_spans)
            data.append(DocumentParagraphQuestion(
                q.question_id, p.doc_id, (p.start, p.end), q.question, p.text, ans, i))

    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)
    test_questions = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(batch_size, True))
    n_questions = len(questions)
    return test_questions, n_questions


def compute_and_dump_official_output(df, savename):
    print("Saving question result")
    answers = {}
    scores = {}
    for q_id, doc_id, start, end, txt, score in df[
        ["question_id", "doc_id", "para_start", "para_end", "text_answer", "predicted_score"]].itertuples(index=False):
        key = q_id
        prev_score = scores.get(key)
        if prev_score is None or prev_score < score:
            scores[key] = score
            answers[key] = txt

    with open(savename, "w", encoding='utf8') as f:
        json.dump(answers, f)


def get_aggregated_df(df, per_document):
    group_by = ["question_id", "doc_id"] if per_document else ["question_id"]
    df.sort_values(group_by + ["rank"], inplace=True)
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    agg_df = pd.DataFrame(list(zip(range(len(em)), em, f1)), columns=["N Paragraphs", "EM", "F1"])
    return agg_df


def test(model, evaluators, datasets: Dict, loader, checkpoint,
         ema=True, aysnc_encoding=None, sample=None, elmo_char_cnn=True) -> Dict[str, Evaluation]:
    print("Setting up model")
    model.set_inputs(list(datasets.values()), loader)

    if aysnc_encoding:
        evaluator_runner = AysncEvaluatorRunner(evaluators, model, aysnc_encoding)
        inputs = evaluator_runner.dequeue_op
    else:
        evaluator_runner = EvaluatorRunner(evaluators, model)
        inputs = model.get_placeholders()
    input_dict = {p: x for p, x in zip(model.get_placeholders(), inputs)}

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        pred = model.get_predictions_for(input_dict)
    evaluator_runner.set_input(pred)

    print("Restoring variables")
    if elmo_char_cnn:
        all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
        vars_to_restore = [x for x in all_vars if x.name not in lm_var_names]
        saver = tf.train.Saver(vars_to_restore)
        sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))
        saver.restore(sess, checkpoint)
    else:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

    if ema:
        # FIXME This is a bit stupid, since we are loading variables twice, but I found it
        # a bit fiddly to load the variables directly....
        ema = tf.train.ExponentialMovingAverage(0)
        reader = tf.train.NewCheckpointReader(checkpoint)
        expected_ema_names = {ema.average_name(x): x for x in tf.trainable_variables()
                              if reader.has_tensor(ema.average_name(x))}
        if len(expected_ema_names) > 0:
            print("Restoring EMA variables")
            saver = tf.train.Saver(expected_ema_names)
            saver.restore(sess, checkpoint)

    tf.get_default_graph().finalize()

    print("Begin evaluation")

    dataset_outputs = {}
    for name, dataset in datasets.items():
        dataset_outputs[name] = evaluator_runner.run_evaluators(sess, dataset, name, sample, {})
    return dataset_outputs


def perform_evaluation(model_name: str,
                       dataset_names: List[str],
                       tokens_per_paragraph: int,
                       filter_type: str,
                       n_processes: int,
                       n_paragraphs: int,
                       batch_size: int,
                       checkpoint: str,
                       no_ema: bool,
                       max_answer_len: int,
                       official_output_path: str,
                       paragraph_output_path: str,
                       aggregated_output_path: str,
                       elmo_char_cnn: bool,
                       n_samples: Union[int, None]
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
    """
    async = True
    corpus_name = 'all'
    per_document = False

    print('Setting Up:')
    model_dir = ModelDir(model_name)
    model = model_dir.get_model()
    dataset = get_multidataset(dataset_names)
    splitter = MergeParagraphs(tokens_per_paragraph)
    para_filter = get_para_filter(filter_type, per_document, n_paragraphs)
    test_questions, n_questions = get_questions(
        per_document, dataset, splitter,
        para_filter, model.preprocessor,
        n_processes, batch_size
    )

    print("Starting eval")
    checkpoint = get_checkpoint(checkpoint, model_dir)
    evaluation = test(
        model,
        [RecordParagraphSpanPrediction(max_answer_len, True)],
        {corpus_name: test_questions},
        ResourceLoader(),
        checkpoint,
        not no_ema,
        async,
        n_samples,
        elmo_char_cnn
    )[corpus_name]

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
    parser.add_argument('model', help='model directory')
    parser.add_argument('-p', '--paragraph_output', type=str,
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument('-ag', '--aggregated_output', type=str,
                        help="Save aggregated results in csv format")
    parser.add_argument('-o', '--official_output', type=str, help="Build an offical output file with the model's"
                                                                  " most confident span for each (question, doc) pair")
    parser.add_argument('--no_ema', default=False, action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--n_processes', type=int, default=8,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-i', '--step', type=int, default=None, help="checkpoint to load, default to latest")
    parser.add_argument('-a', '--async', type=int, default=10)
    parser.add_argument('-t', '--tokens', type=int, default=400,
                        help="Max tokens per a paragraph")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=15,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-f', '--filter', type=str, default=None, choices=["tfidf", "truncate", "linear"],
                        help="How to select paragraphs")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help="Batch size, larger sizes might be faster but wll take more memory")
    parser.add_argument('--max_answer_len', type=int, default=50,
                        help="Max answer span to select")
    parser.add_argument('-d', '--datasets', nargs='+', default=['squad'], help='Which datasets to compute')
    parser.add_argument('-c', '--elmo_character_cnn', action='store_true', dest='elmo_character_cnn',
                        help='Use Elmo char CNN - if false, uses precomputed token representations')
    parser.add_argument('-no_c', '--no_elmo_character_cnn', action='store_false', dest='elmo_character_cnn')
    parser.add_argument('-s', '--samples', type=int, default=None, help='Number of samples to run, defaults to all')

    parser.set_defaults(elmo_character_cnn=True)
    args = parser.parse_args()

    perform_evaluation(
        model_name=args.model,
        dataset_names=args.datasets,
        tokens_per_paragraph=args.tokens,
        filter_type=args.filter,
        n_processes=args.n_processes,
        n_paragraphs=args.n_paragraphs,
        batch_size=args.batch_size,
        checkpoint=args.step,
        no_ema=args.no_ema,
        max_answer_len=args.max_answer_len,
        official_output_path=args.official_output,
        paragraph_output_path=args.paragraph_output,
        aggregated_output_path=args.aggregated_output,
        elmo_char_cnn=args.elmo_character_cnn,
        n_samples=args.samples
    )


if __name__ == "__main__":
    main()
