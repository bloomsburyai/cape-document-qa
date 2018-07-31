import os
from nltk.downloader import download
from logging import info
from cape_document_qa.download_and_extract import download_and_extract, download_file
from cape_document_qa.cape_document_qa_settings import GLOVE_EMBEDDINGS_URL, LM_URL
from cape_document_qa.cape_config import VEC_DIR, SQUAD_SERVER, SQUAD_SOURCE_DIR, TRIVIAQA_SERVER, TRIVIA_QA, LM_DIR


def training_downloads():
    # NLTK:
    info("Downloading (if necessary) NLTK resources:")
    download('punkt')
    download('stopwords')

    # Glove:
    info('Downloading Glove Embeddings:')
    download_and_extract(GLOVE_EMBEDDINGS_URL, VEC_DIR)

    # Squad:
    info('Downloading Squad:')
    download_file(SQUAD_SERVER + '/train-v1.1.json', SQUAD_SOURCE_DIR)
    download_file(SQUAD_SERVER + '/dev-v1.1.json', SQUAD_SOURCE_DIR)

    # TriviaQA:
    info('Downloading TriviaQA:')
    if not os.path.exists(TRIVIA_QA):
        os.makedirs(TRIVIA_QA)
    download_and_extract(TRIVIAQA_SERVER + 'triviaqa-rc.tar.gz', TRIVIA_QA)

    # LM:
    info('Downloading LM:')
    if not os.path.exists(LM_DIR):
        os.makedirs(LM_DIR)
    download_and_extract(LM_URL, LM_DIR)


if __name__ == '__main__':
    training_downloads()