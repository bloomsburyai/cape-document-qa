import os
from nltk.downloader import download
from logging import info
from cape_document_qa.download_and_extract import download_and_extract
from cape_document_qa.cape_document_qa_settings import MODEL_FOLDER, MODEL_URL, MODELS_FOLDER, MODEL_MB_SIZE, \
    GLOVE_EMBEDDINGS_URL, DOWNLOAD_ALL_GLOVE_EMBEDDINGS

glove_filepath = os.path.join(MODEL_FOLDER, 'glove.840B.300d.txt')
if not os.path.isfile(os.path.join(MODEL_FOLDER, 'model.pkl')) or \
        not os.path.isfile(glove_filepath) or \
        (
                DOWNLOAD_ALL_GLOVE_EMBEDDINGS and os.path.getsize(glove_filepath) / 1e6 < 2e3
                # less than 2 GBs-> we only have the top X embeddings
        ):
    # Downloading NLTK dependencies
    info("Downloading (if necessary) NLTK ressources:")
    download('punkt')
    download('stopwords')
    info('Downloading default model with top X Glove embeddings:')
    download_and_extract(MODEL_URL, MODELS_FOLDER, total_mb_size=MODEL_MB_SIZE)
    if DOWNLOAD_ALL_GLOVE_EMBEDDINGS:
        info('Downloading complete Glove Embeddings:')
        download_and_extract(GLOVE_EMBEDDINGS_URL, MODEL_FOLDER)
