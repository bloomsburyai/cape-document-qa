from cape_document_qa import patches
import json
from tqdm import tqdm
from docqa.elmo.lm_model import dump_token_embeddings
from docqa.triviaqa.build_span_corpus import TriviaQaSpanCorpus
from cape_document_qa import cape_config
from typing import List


def get_dataset_vocab_count(dataset_name):
    corpus = TriviaQaSpanCorpus(dataset_name)
    vocab_counter = {}

    def _add_to_vocab_counter(questions):
        for question in questions:
            for w in question.question:
                vocab_counter[w] = vocab_counter.get(w, 0) + 1

    _add_to_vocab_counter(corpus.get_train())
    _add_to_vocab_counter(corpus.get_dev())
    _add_to_vocab_counter(corpus.get_test())

    for document_id in tqdm(
            corpus.evidence.list_documents(),
            desc='Counting {} Vocab'.format(dataset_name),
            ncols=80
    ):
        for w in corpus.evidence.get_document(document_id, flat=True):
            vocab_counter[w] = vocab_counter.get(w, 0) + 1

    return vocab_counter


def get_elmo_vocab(dataset_names, min_word_count):

    vocab_counter = {}
    for dataset_name in dataset_names:
        dataset_vocab_counter = get_dataset_vocab_count(dataset_name)
        for w, c in dataset_vocab_counter.items():
            vocab_counter[w] = vocab_counter.get(w, 0) + c

    vocab = {w for w, c in vocab_counter.items() if c > min_word_count}
    special_chars = ['<S>', '</S>', '<UNK>']
    return special_chars + sorted(list(vocab), key=lambda x: vocab_counter[x], reverse=True)


def serialize_elmo_vocab(vocab):
    with open(cape_config.LM_VOCAB, 'w', encoding='utf8') as f:
        for v in vocab:
            f.write(v + '\n')


def serialize_elmo_model_options(num_tokens):
    with open(cape_config.LM_INITIAL_OPTIONS, encoding='utf8') as f:
        bilm_model_opts = json.load(f)
        bilm_model_opts['n_tokens_vocab'] = num_tokens + 1
    with open(cape_config.LM_OPTIONS, 'w', encoding='utf8') as f:
        json.dump(bilm_model_opts, f)


def compute_and_dump_token_vectors(datasets: List, min_word_count: int):
    """Precompute Elmo Token Vectors before training using the ELMO LM ConvNet.
    The tokens are saved in lm directory in the data directory

    :param datasets: list of names of the datasets as they appear in the data directory, 'e.g.' ['wiki',]
    :param min_word_count: mininmum unigram count of a token for a vector to be computed
    """
    elmo_vocab = get_elmo_vocab(datasets, min_word_count)
    serialize_elmo_model_options(len(elmo_vocab))
    serialize_elmo_vocab(elmo_vocab)
    print('Calculating Token Embeddings')
    dump_token_embeddings(
        cape_config.LM_VOCAB,
        cape_config.LM_OPTIONS,
        cape_config.LM_WEIGHTS,
        cape_config.LM_TOKEN_WEIGHTS
    )


if __name__ == '__main__':
    min_word_count = 100
    datasets = ['wiki']
    compute_and_dump_token_vectors(datasets, min_word_count)
