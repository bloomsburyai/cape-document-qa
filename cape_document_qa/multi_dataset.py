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

from docqa.triviaqa.read_data import TriviaQaQuestion
from os.path import join
from typing import List, Optional
from docqa.utils import ResourceLoader


class MultiEvidence(object):

    def __init__(self, names, evidences):
        self.evidence_dict = {n: e for n, e in zip(names, evidences)}
        self.file_id_map = {n + '/' + i: p for n, e in self.evidence_dict.items() for i, p in e.file_id_map.items()}
        self.evidence_doc_map = {n + '/' + i: n for n, e in self.evidence_dict.items() for i in e.file_id_map.keys()}
        self.default_evidence = evidences[0]

    def get_vocab(self):
        vocab = set()
        for e in self.evidence_dict.values():
            vocab.update(e.get_vocab())
        raise NotImplementedError('Dont trust those who ask for vocab')
        # return vocab

    def load_word_vectors(self, vec_name):
        raise NotImplementedError()

    def list_documents(self):
        return list(self.file_id_map.keys())

    def get_document(self, doc_id, n_tokens=None, flat=False):
        evidence = self.evidence_dict.get(self.evidence_doc_map.get(doc_id), self.default_evidence)
        try:
            file_id = join(evidence.directory, self.file_id_map.get(doc_id) + ".txt")
        except:
            print(doc_id)
        return evidence.get_document_from_file_path(file_id, n_tokens=n_tokens, flat=flat)


class MultiDataset(object):

    def __init__(self, datasets):
        self.datasets = datasets
        self.corpus_name = '|'.join([d.name for d in self.datasets])
        self.evidence = MultiEvidence(*zip(*[(dataset.name, dataset.evidence) for dataset in self.datasets]))
        self.train_doc_ids_reset = False
        self.dev_doc_ids_reset = False
        self.test_doc_ids_reset = False
        self.verified_doc_ids_reset = False


    @staticmethod
    def _reset_doc_ids(prefix, questions):
        for q in questions:
            for d in q.all_docs:
                new_doc_id = prefix + '/' + d.doc_id
                if 'title' in d.__slots__:
                    d.title = new_doc_id
                if 'url' in d.__slots__:
                    d.url = new_doc_id
                if d.doc_id != new_doc_id:
                    raise Exception(d.__slots__)
                assert d.doc_id == new_doc_id, (d.doc_id, new_doc_id, d)

    def _get_train_data(self, dataset):
        train_questions = dataset.get_train()
        MultiDataset._reset_doc_ids(dataset.name, train_questions)
        return train_questions

    def _get_dev_data(self, dataset):
        dev_questions = dataset.get_dev()
        MultiDataset._reset_doc_ids(dataset.name, dev_questions)
        return dev_questions

    def _get_test_data(self, dataset):
        test_questions = dataset.get_test()
        MultiDataset._reset_doc_ids(dataset.name, test_questions)
        return test_questions

    def _get_verified_data(self, dataset):
        verified_questions = dataset.get_verified()
        MultiDataset._reset_doc_ids(dataset.name, verified_questions)
        return verified_questions

    def get_train(self) -> List[TriviaQaQuestion]:
        return [q for dataset in self.datasets for q in self._get_train_data(dataset)]

    def get_dev(self) -> List[TriviaQaQuestion]:
        return [q for dataset in self.datasets for q in self._get_dev_data(dataset)]

    def get_test(self) -> List[TriviaQaQuestion]:
        return [q for dataset in self.datasets for q in self._get_test_data(dataset)]

    def get_verified(self) -> Optional[List[TriviaQaQuestion]]:
        return [q for dataset in self.datasets for q in self._get_verified_data(dataset)]

    def get_resource_loader(self):
        return ResourceLoader()

    @property
    def name(self):
        return self.corpus_name
