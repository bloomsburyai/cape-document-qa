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
import numpy as np
from cape_document_qa.cape_config import PREPRO_DATASET_DIR, PREPRO_EVIDENCE_DIR, SQUAD_SOURCE_DIR
import pickle
from os import makedirs
from os.path import join, exists
from docqa.squad.build_squad_dataset import parse_squad_data
from docqa.triviaqa.read_data import TriviaQaQuestion, TagMeEntityDoc, FreeForm
from tqdm import tqdm
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
import json
from typing import Dict


def get_out_dir(dataset_name):
    return join(PREPRO_DATASET_DIR, dataset_name)


def get_doc_savename(dirpath, doc_id):
    return join(dirpath, doc_id.replace('/', ''))


def get_questions_savename(dataset_name, fold):
    return join(get_out_dir(dataset_name), fold + ".pkl")


def dump_paragraph(paragraph):
    return "\n\n".join("\n".join(" ".join(sent) for sent in para) for para in [paragraph.text])


def squad_q2triviaqa_q(question):
    doc_id = question.question_id
    question_id = question.question_id

    answer_spans = np.unique(question.answer.answer_spans, axis=0)
    answer_texts = list(set(question.answer.answer_text))

    answer = FreeForm(
        value=answer_texts[0],
        normalized_value=answer_texts[0].lower(),
        aliases=answer_texts,
        normalized_aliases=answer_texts,
        human_answers=None
    )
    doc = TagMeEntityDoc(1., 1., doc_id)
    doc.answer_spans = answer_spans
    return TriviaQaQuestion(question.words, question_id, answer, [doc], [])


def prepro_squad_fold(name, fold, squad_file_paths):
    tokenizer = NltkAndPunctTokenizer()
    dataset_evidence_dir = join(PREPRO_EVIDENCE_DIR, name)

    if not exists(dataset_evidence_dir):
        makedirs(dataset_evidence_dir)

    voc = set()
    squad_docs = [
        d for squad_file_path in squad_file_paths
        for d in parse_squad_data(squad_file_path, fold, tokenizer)
    ]
    questions = []
    file_map = {}

    for document in tqdm(squad_docs, desc=fold, ncols=80):
        for paragraph in document.paragraphs:
            for question in paragraph.questions:
                doc_id = question.question_id
                doc_savename = get_doc_savename(dataset_evidence_dir, doc_id)
                trivia_q = squad_q2triviaqa_q(question)

                with open(doc_savename + '.txt', 'w', encoding='utf8') as f:
                    f.write(dump_paragraph(paragraph))

                words = {w for sent in paragraph.text for w in sent}
                voc.update(words)

                file_map[doc_id] = doc_savename
                questions.append(trivia_q)

    questions_savename = get_questions_savename(name, fold)
    with open(questions_savename, "wb") as f:
        pickle.dump(questions, f)

    return voc, file_map


def preprocess_squad_dataset(name: str, fold_dict: Dict):
    """Preprocess a squad dataset for training. Creates entries in the dataset directory/triviaqa directory
    and adds entries in the dataset directory/triviaqa/evidence directories for questions and documents respectively

    :param name: The name of the dataset  (e.g. Squad)
    :param fold_dict: keys: name of the fold, values: list of paths to the json of that fold,
    e.g. {'train': ['path/to/train.json'], 'dev': ['path/to/dev.json'], 'test': ['path/to/test.json']]}
    """

    print('Preprocessing Squad Dataset: {}'.format(name))
    if not exists(get_out_dir(name)):
        makedirs(get_out_dir(name))

    voc, file_map = set(), {}
    for fold, squad_file_paths in fold_dict.items():
        fold_voc, fold_file_map = prepro_squad_fold(name, fold, squad_file_paths)
        voc.update(fold_voc)
        for k, v in fold_file_map.items():
            file_map[k] = v

    print("Dumping file mapping")
    with open(join(get_out_dir(name), "file_map.json"), "w", encoding='utf8') as f:
        json.dump(file_map, f)

    print("Dumping vocab mapping")
    with open(join(get_out_dir(name), "vocab.txt"), "w", encoding='utf8') as f:
        for word in sorted(voc):
            f.write(word)
            f.write("\n")


def squad_dict():
    return 'squad', {
        'train': [join(SQUAD_SOURCE_DIR, 'train-v1.1.json')],
        'dev': [join(SQUAD_SOURCE_DIR, 'dev-v1.1.json')],
        'test': [join(SQUAD_SOURCE_DIR, 'dev-v1.1.json')],
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Perform dataset preprocessing for Squad models.")
    parser.add_argument('-n', '--dataset_name', type=str, default=None, dest='title', help='name of dataset to preprocess')
    parser.add_argument('-tr', '--train_files', nargs='+', default=[], help='Which datasets to compute')
    parser.add_argument('-dv', '--dev_files', nargs='+', default=[], help='Which datasets to compute')
    parser.add_argument('-ts', '--test_files', nargs='+', default=[], help='Which datasets to compute')
    args = parser.parse_args()

    if args.title is None:
        title, fold_dict = squad_dict()
    else:
        title = args.title
        fold_dict = {'train': args.train_files, 'dev': args.dev_files, 'test': args.test_files}
    preprocess_squad_dataset(title, fold_dict)
