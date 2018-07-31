from os.path import join, expanduser, dirname
from docqa import config

"""
Global config options
"""
CORPUS_DIR = join(dirname(dirname(__file__)), "data")
PREPRO_DATASET_DIR = join(CORPUS_DIR, 'triviaqa')
PREPRO_EVIDENCE_DIR = join(PREPRO_DATASET_DIR, 'evidence')
PREPRO_VOCAB_PATH = join(PREPRO_DATASET_DIR, 'vocab.txt')

ROOT_DATA_DIR = join(expanduser("~"), "data")
VEC_DIR = join(ROOT_DATA_DIR, "glove")

SQUAD_SOURCE_DIR = join(ROOT_DATA_DIR, "bai_annotated_question_answering_unique_ids_better_lengths")
SQUAD_TRAIN = join(SQUAD_SOURCE_DIR, "train-v1.1.json")
SQUAD_DEV = join(SQUAD_SOURCE_DIR, "dev-v1.1.json")

TRIVIA_QA = join(ROOT_DATA_DIR, "triviaqa-rc")
TRIVIA_QA_UNFILTERED = join(ROOT_DATA_DIR, "triviaqa-unfiltered")
TRIVIA_QA_EVIDENCE = join(TRIVIA_QA, 'evidence')
TRIVIA_QA_WIKI_EVIDENCE = join(TRIVIA_QA_EVIDENCE, 'wikipedia')
TRIVIA_QA_WEB_EVIDENCE = join(TRIVIA_QA_EVIDENCE, 'web')

LM_DIR = join(CORPUS_DIR, "lm")
LM_INITIAL_OPTIONS = join(LM_DIR, "elmo_initial_options.json")
LM_OPTIONS = join(LM_DIR, "elmo_options.json")
LM_WEIGHTS = join(LM_DIR, 'elmo_weights.hdf5')
LM_TOKEN_WEIGHTS = join(LM_DIR, 'elmo_token_vectors.hdf5')
LM_VOCAB = join(LM_DIR, 'vocab.txt')


config.CORPUS_DIR = CORPUS_DIR
config.VEC_DIR = VEC_DIR
config.SQUAD_SOURCE_DIR = SQUAD_SOURCE_DIR
config.SQUAD_TRAIN = SQUAD_TRAIN
config.SQUAD_DEV = SQUAD_DEV
config.TRIVIA_QA = TRIVIA_QA
config.TRIVIA_QA_UNFILTERED = TRIVIA_QA_UNFILTERED
config.LM_DIR = LM_DIR
config.LM_OPTIONS = LM_OPTIONS
config.LM_WEIGHTS = LM_WEIGHTS
config.LM_VOCAB = LM_VOCAB


"""
Download configs
"""

CAPE_DATA_SERVER = "https://github.com/bloogram/cape-document-qa/releases/download/"
CAPE_DATA_RELEASE = "v0.1.0"

GLOVE_SERVER = 'http://nlp.stanford.edu/data/'
SQUAD_SERVER = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
TRIVIAQA_SERVER = 'http://nlp.cs.washington.edu/triviaqa/data/'

PRODUCTION_MODEL_NAME = 'production_ready_model'


"""
Training configs
"""


class TrainConfig(object):
    trivia_qa_mode = 'shared-norm'
    n_epochs = 80
    train_batch_size = 30
    test_batch_size = 60
    oversample = [1] * 2
    learning_rate = 1.
    optimizer = 'Adadelta'
    n_tokens = 400
    ema = 0.999
    max_checkpoints_to_keep = 2
    async_encoding = 10
    log_period=30
    eval_period = 1800
    save_period = 1800
    num_paragraphs = 16
    dim = 120
    char_th = 14
    l2 = 0
    top_layer_only = False
    recurrent_stdev = 0.05
    elmo_mode = 'input'
    word_vectors = "glove.840B.300d"
    learn_unk_vector = True
    max_batch_size = 128
    lm_layernorm = False
    char_dim = 20
    var_dropout = 0.8
    elmo_dropout = 0.5