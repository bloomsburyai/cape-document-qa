from docqa import config
from cape_document_qa import cape_config

config.CORPUS_DIR = cape_config.CORPUS_DIR
config.VEC_DIR = cape_config.VEC_DIR
config.SQUAD_SOURCE_DIR = cape_config.SQUAD_SOURCE_DIR
config.SQUAD_TRAIN = cape_config.SQUAD_TRAIN
config.SQUAD_DEV = cape_config.SQUAD_DEV
config.TRIVIA_QA = cape_config.TRIVIA_QA
config.TRIVIA_QA_UNFILTERED = cape_config.TRIVIA_QA_UNFILTERED
config.LM_DIR = cape_config.LM_DIR
config.LM_OPTIONS = cape_config.LM_OPTIONS
config.LM_WEIGHTS = cape_config.LM_WEIGHTS
config.LM_VOCAB = cape_config.LM_VOCAB