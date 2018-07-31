from cape_document_qa import patches
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.utils import ResourceLoader
from docqa.data_processing.word_vectors import load_word_vectors
from cape_machine_reader.cape_machine_reader_model import CapeMachineReaderModelInterface
import tensorflow as tf
import numpy as np
import pickle
from os.path import join
from cape_document_qa.cape_document_qa_settings import MODEL_FOLDER

vocab_to_ignore = {'<S>', '</S>', '<UNK>', '!!!MAXTERMID'}


class CapeDocQAMachineReaderConfig:

    def __init__(self,
                 model_pickle_file,
                 model_checkpoint_file,
                 lm_weights_file,
                 lm_token_weights_file,
                 lm_options_file,
                 vocab_file,
                 word_vector_file
                 ):
        self.model_pickle_file = model_pickle_file
        self.checkpoint_file = model_checkpoint_file
        self.lm_weights_file = lm_weights_file
        self.lm_token_weights_file = lm_token_weights_file
        self.lm_options_file = lm_options_file
        self.vocab_file = vocab_file
        self.word_vector_file = word_vector_file


class CapeDocQAMachineReaderModel(CapeMachineReaderModelInterface):

    def __init__(self, machine_reader_config):
        self.tokenizer = NltkAndPunctTokenizer()
        self.config = machine_reader_config
        self.model = self._load_model()
        self.sess = tf.Session()
        self.start_logits, self.end_logits, self.context_rep = self._build_model()
        self._initialize()

    def _load_model(self):
        with open(self.config.model_pickle_file, 'rb') as f:
            model = pickle.load(f)

        model.lm_model.weight_file = self.config.lm_weights_file
        model.lm_model.lm_vocab_file = self.config.vocab_file
        model.lm_model.embed_weights_file = self.config.lm_token_weights_file
        model.lm_model.options_file = self.config.lm_options_file
        return model

    def _build_model(self):
        vocab_to_init_with = {
            line.strip() for line in open(self.config.vocab_file, encoding="utf-8")
            if line.strip() not in vocab_to_ignore
        }
        self.model.word_embed.vec_name = self.config.word_vector_file
        with self.sess.as_default():
            self.model.set_input_spec(
                ParagraphAndQuestionSpec(None, None, None, 14), vocab_to_init_with,
                word_vec_loader=ResourceLoader(load_vec_fn=lambda x, y: load_word_vectors(x, y, is_path=True))
            )
            pred = self.model.get_production_predictions_for({x: x for x in self.model.get_placeholders()})
        return pred.start_logits, pred.end_logits, self.model.context_rep

    def _initialize(self):
        all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
        vars_to_restore = [x for x in all_vars if x.name not in lm_var_names]
        saver = tf.train.Saver(vars_to_restore)
        saver.restore(self.sess, self.config.checkpoint_file)
        self.sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize_paragraph_flat(text)
        spans = self.tokenizer.convert_to_spans(text, [tokens])[0]
        return tokens, spans

    def get_document_embedding(self, text):
        document_tokens, _ = self.tokenize(text)
        test_question = ParagraphAndQuestion(document_tokens, ['dummy', 'question'], None, "cape_question",
                                             'cape_document')
        feed = self.model.encode([test_question], False, cached_doc=None)
        return self.sess.run(self.model.context_rep, feed_dict=feed)[0]

    def get_logits(self, question, document_embedding):
        question_tokens, _ = self.tokenize(question)
        n_words = document_embedding.shape[0]
        dummy_document = ['dummy'] * n_words
        test_question = ParagraphAndQuestion(dummy_document, question_tokens, None, "cape_question", 'cape_document')
        feed = self.model.encode([test_question], False, cached_doc=document_embedding[np.newaxis, :, :])
        start_logits, end_logits = self.sess.run([self.start_logits, self.end_logits], feed_dict=feed)
        return start_logits[0][:n_words], end_logits[0][:n_words]


def get_production_model_config():
    return CapeDocQAMachineReaderConfig(
        model_pickle_file=join(MODEL_FOLDER, 'model.pkl'),
        model_checkpoint_file=join(MODEL_FOLDER, 'save', 'checkpoint-123456789'),
        lm_weights_file=join(MODEL_FOLDER, 'elmo_weights.hdf5'),
        lm_token_weights_file=None,
        lm_options_file=join(MODEL_FOLDER, 'elmo_options.json'),
        vocab_file=join(MODEL_FOLDER, 'vocab.txt'),
        word_vector_file=join(MODEL_FOLDER, 'glove.840B.300d')
    )


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Run model evaluation using Cape's model decoder")
    parser.add_argument('-m', '--model', default=None, help='path to model directory')
    args = parser.parse_args()
    if args.model is not None:
        conf = CapeDocQAMachineReaderConfig(
            model_pickle_file=join(args.model, 'model.pkl'),
            model_checkpoint_file=join(args.model, 'save', 'checkpoint-123456789'),
            lm_weights_file=join(args.model, 'elmo_weights.hdf5'),
            lm_token_weights_file=None,
            lm_options_file=join(args.model, 'elmo_options.json'),
            vocab_file=join(args.model, 'vocab.txt'),
            word_vector_file=join(args.model, 'glove.840B.300d')
        )
    else:
        conf = get_production_model_config()

    machine_reader = CapeDocQAMachineReaderModel(conf)

    context = '''"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."'''
    question = "Which NFL team represented the AFC at Super Bowl 50?"

    num_trials = 10
    t1 = time.time()
    for i in range(num_trials):
        print('Iteration {0}'.format(i))
        doc_embedding = machine_reader.get_document_embedding(context)
        start_logits, end_logits = machine_reader.get_logits(question, doc_embedding)
        toks, offs = machine_reader.tokenize(context)
        st, en = offs[np.argmax(start_logits)], offs[np.argmax(end_logits)]
        print(context[st[0]: en[1]])

    t2 = time.time()
    print('full reading takes {} seconds'.format((t2 - t1) / num_trials))

    for i in range(num_trials):
        print('Iteration {0}'.format(i))
        start_logits, end_logits = machine_reader.get_logits(question, doc_embedding)
        toks, offs = machine_reader.tokenize(context)
        st, en = offs[np.argmax(start_logits)], offs[np.argmax(end_logits)]
        print(context[st[0]: en[1]])
    t3 = time.time()
    print('Cached reading takes {} seconds'.format((t3 - t2) / num_trials))
#
