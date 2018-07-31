import h5py
from docqa.elmo.data import UnicodeCharsVocabulary, Batcher
from docqa.elmo.lm_model import BidirectionalLanguageModel
import numpy as np
import tensorflow as tf
import json
from docqa.elmo import lm_model
from tqdm import tqdm


def dump_token_embeddings(vocab_file, options_file, weight_file, outfile):
    """
    Given an input vocabulary file, dump all the token embeddings to the
    outfile.  The result can be used as the embedding_weight_file when
    constructing a BidirectionalLanguageModel.

    Patched to print progress
    """
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = Batcher(vocab_file, max_word_length)
    print('Computing {} LM token vectors'.format(vocab.size))

    ids_placeholder = tf.placeholder(
        'int32',
        shape=(None, None, max_word_length)
    )
    print('Building Language model')
    model = BidirectionalLanguageModel(
        options_file, weight_file, ids_placeholder
    )

    embedding_op = model.get_ops()['token_embeddings']

    n_tokens = vocab.size
    embed_dim = int(embedding_op.shape[2])

    embeddings = np.zeros((n_tokens, embed_dim), dtype=lm_model.DTYPE)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in tqdm(range(n_tokens), desc='Computing LM token embeddings', ncols=80):
            token = vocab.id_to_word(k)
            char_ids = batcher.batch_sentences([[token]])[0, 1, :].reshape(
                1, 1, -1)
            embeddings[k, :] = sess.run(
                embedding_op, feed_dict={ids_placeholder: char_ids}
            )

    with h5py.File(outfile, 'w') as fout:
        fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )


lm_model.dump_token_embeddings = dump_token_embeddings

