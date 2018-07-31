import numpy as np
from cape_document_qa.cape_docqa_machine_reader import get_production_model_config,CapeDocQAMachineReaderModel
from cape_machine_reader.cape_machine_reader_model import CapeMachineReaderModelInterface
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
import hashlib

from pytest import fixture


class RandomMachineReaderModel(CapeMachineReaderModelInterface):

    def __init__(self, _):
        self.tokenizer = NltkAndPunctTokenizer()

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize_paragraph_flat(text)
        spans = self.tokenizer.convert_to_spans(text, [tokens])[0]
        return tokens, spans

    def get_document_embedding(self, text):
        np.random.seed(int(hashlib.sha1(text.encode()).hexdigest(), 16) % 10 ** 8)
        document_tokens, _ = self.tokenize(text)
        return np.random.random((len(document_tokens), 240))

    def get_logits(self, question, document_embedding):
        question_tokens, _ = self.tokenize(question)
        n_words = document_embedding.shape[0]
        qseed = int(hashlib.sha1(question.encode()).hexdigest(), 16) % 10 ** 8
        dseed = int(np.sum(document_embedding) * 10 ** 6) % 10 ** 8
        np.random.seed(dseed + qseed)
        start_logits = np.random.random(n_words)
        off = np.random.randint(1, 5)
        end_logits = np.concatenate([np.zeros(off) + np.min(start_logits), start_logits[off:]])
        return start_logits[:n_words], end_logits[:n_words]


@fixture
def context():
    return '''"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."'''


@fixture
def question():
    return "Which NFL team represented the AFC at Super Bowl 50?"


@fixture
def answer():
    return 'Denver Broncos'


def test_machine_reader_e2e(question, context, answer):
    conf = get_production_model_config()
    machine_reader = CapeDocQAMachineReaderModel(conf)
    doc_embedding = machine_reader.get_document_embedding(context)
    start_logits, end_logits = machine_reader.get_logits(question, doc_embedding)
    toks, offs = machine_reader.tokenize(context)
    st, en = offs[np.argmax(start_logits)], offs[np.argmax(end_logits)]
    mr_answer = context[st[0]: en[1]]
    assert answer == mr_answer
