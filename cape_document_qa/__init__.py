import os
from nltk.downloader import download
from logging import info
from cape_document_qa.download_and_extract import download_and_extract
from cape_document_qa.cape_document_qa_settings import MODEL_FOLDER, MODEL_URL, MODELS_FOLDER, MODEL_MB_SIZE, \
    GLOVE_EMBEDDINGS_URL

if not os.path.isfile(os.path.join(MODEL_FOLDER, 'model.pkl')) or \
        not os.path.isfile(os.path.join(MODEL_FOLDER, 'glove.840B.300d.txt')):
    # Downloading NLTK dependencies
    info("Downloading (if necessary) NLTK ressources:")
    download('punkt')
    download('stopwords')
    info('Downloading default model:')
    download_and_extract(MODEL_URL, MODELS_FOLDER, total_mb_size=MODEL_MB_SIZE)
    info('Downloading Glove Embeddings:')
    download_and_extract(GLOVE_EMBEDDINGS_URL, MODEL_FOLDER)
