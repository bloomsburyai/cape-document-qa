from cape_document_qa import patches
from os.path import join
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from cape_document_qa.preprocessing.cape_preprocess_squad import preprocess_squad_dataset
from cape_document_qa.preprocessing.compute_elmo_vectors import compute_and_dump_token_vectors, get_dataset_vocab_count
from docqa.triviaqa.evidence_corpus import build_tokenized_corpus
from docqa.triviaqa.build_span_corpus import build_web_corpus, build_wiki_corpus
from cape_document_qa.cape_config import SQUAD_SOURCE_DIR, TRIVIA_QA_EVIDENCE, PREPRO_EVIDENCE_DIR, PREPRO_VOCAB_PATH


def squad_prepro(squad_dir, datasets):
    print('Preprocessing Squad Files:')
    for title, fold_dict in datasets.items():
        fd = {f: [join(squad_dir, fi) for fi in fold_dict[f]] for f in ['train', 'dev', 'test']}
        preprocess_squad_dataset(title, fd)


def triviaqa_prepro(wiki_only, n_processes):
    print('Tokenizing {} corpus:'.format('wiki' if wiki_only else 'wiki and web'))
    build_tokenized_corpus(
        TRIVIA_QA_EVIDENCE,
        NltkAndPunctTokenizer(),
        PREPRO_EVIDENCE_DIR,
        n_processes=n_processes,
        wiki_only=wiki_only,
    )
    print('Preparing wiki corpus:')
    build_wiki_corpus(n_processes)
    if not wiki_only:
        print('Preparing web corpus:')
        build_web_corpus(n_processes)


def vocab_prepro(datasets):
    voc = set()
    for d in datasets:
        print('Collecting Vocab for {}'.format(d))
        ds_vocab = get_dataset_vocab_count(d).keys()
        voc.update(ds_vocab)
        print('{} Vocab Size: {}'.format(d, len(ds_vocab)))

    print('total Vocab Size: {}'.format(len(voc)))
    with open(PREPRO_VOCAB_PATH, 'w', encoding='utf8') as f:
        for v in voc:
            f.write(v + '\n')


def full_preprocessing_pipeline(squad_datasets, triviaqa_datasets, lm_min_word_count, n_procs):
    datasets = list(triviaqa_datasets) + list(squad_datasets.keys())
    squad_prepro(SQUAD_SOURCE_DIR, squad_datasets)
    if ('wiki' in datasets) or ('web' in datasets):
        triviaqa_prepro('web' not in datasets, n_procs)
    vocab_prepro(datasets)
    compute_and_dump_token_vectors(datasets, lm_min_word_count)


def default_dataset_dict():
    return {
        'triviaqa_datasets': ['wiki', 'web'],
        'squad_datasets': {
            'squad': {
                'train': ['train-v1.1.json'],
                'dev': ['dev-v1.1.json'],
                'test': ['dev-v1.1.json'],
            }
        }
    }


def main(datasets_dict, n_procs=8, elmo_min_word_count=100):
    """Preprocess datasets for training

    :param datasets_dict: a dictionary specifying the dataset folds and filenames
        to preprocess, see `default_dataset_dict()` for an example
    :param n_procs: number of processes to use
    :param elmo_min_word_count: minimum number of occurrences of a word before an elmo vector is computed for it
    """
    full_preprocessing_pipeline(
        squad_datasets=datasets_dict['squad_datasets'],
        triviaqa_datasets=datasets_dict['triviaqa_datasets'],
        lm_min_word_count=elmo_min_word_count,
        n_procs=n_procs
    )


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Perform dataset preprocessing for training models. See Readme for details")
    parser.add_argument('-d', '--dataset_dict', default='', type=str, dest='dataset_dict', help='path to dataset_dict')
    parser.add_argument('-p', '--n_processes', type=int, default=8, help="Number of processes for preprocessing")
    parser.add_argument('-c', '--elmo_min_word_count', type=int, default=100,
                        help="minimum number of occurrences of a word before an elmo vector is computed for it")
    args = parser.parse_args()

    if args.dataset_dict == '':
        dataset_dict = default_dataset_dict()
    else:
        dataset_dict = json.load(open(args.dataset_dict))

    main(dataset_dict, n_procs=args.n_processes, elmo_min_word_count=args.elmo_min_word_count)
