from cape_document_qa import patches
from cape_document_qa import cape_config
from cape_document_qa.cape_config import LM_VOCAB, LM_OPTIONS, LM_WEIGHTS, LM_TOKEN_WEIGHTS, TrainConfig
import argparse
from datetime import datetime
from typing import Optional, Dict
import json
from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import ShallowOpenWebRanker
from docqa.triviaqa.build_span_corpus import TriviaQaSpanCorpus
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion
from docqa.data_processing.document_splitter import MergeParagraphs
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.encoder import DocumentAndQuestionEncoder, GroupedSpanAnswerEncoder
from docqa.nn.attention import BiAttention, StaticAttentionSelf
from docqa.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from docqa.nn.layers import NullBiMapper, SequenceMapperSeq, Conv1d, FullyConnected, \
    ChainBiMapper, ConcatWithProduct, ResidualLayer, VariationalDropoutLayer, MaxPool, MapperSeq, DropoutLayer
from docqa.nn.recurrent_layers import CudnnGru, BiRecurrentMapper, CompatGruCellSpec, TruncatedNormal
from docqa.nn.similarity_layers import TriLinear
from docqa.nn.span_prediction import BoundsPredictor, IndependentBoundsGrouped
from docqa.text_preprocessor import WithIndicators, TextPreprocessor
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.elmo.elmo import ElmoLayer
from cape_document_qa.cape_models.cape_lm_qa_models import CapeAttentionWithElmo
from cape_document_qa.multi_dataset import MultiDataset
from docqa.elmo.lm_model import LanguageModel
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator


def build_model(preprocess: Optional[TextPreprocessor], train_config, use_cudnn=False):
    if use_cudnn:
        print('Using Cuddn:')
        recurrent_layer = CudnnGru(train_config.dim, w_init=TruncatedNormal(stddev=train_config.recurrent_stdev))
    else:
        recurrent_layer = BiRecurrentMapper(CompatGruCellSpec(train_config.dim))

    lm_reduce = MapperSeq(
        ElmoLayer(
            train_config.l2,
            layer_norm=train_config.lm_layernorm,
            top_layer_only=train_config.top_layer_only
        ),
        DropoutLayer(train_config.elmo_dropout),
    )

    answer_encoder = GroupedSpanAnswerEncoder()
    predictor = BoundsPredictor(
        ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ),
        span_predictor=IndependentBoundsGrouped(aggregate="sum")
    )
    word_embed = FixedWordEmbedder(
        vec_name=train_config.word_vectors,
        word_vec_init_scale=0,
        learn_unk=train_config.learn_unk_vector,
        cpu=True
    )
    char_embed = CharWordEmbedder(
        LearnedCharEmbedder(
            word_size_th=14,
            char_th=train_config.char_th,
            char_dim=train_config.char_dim,
            init_scale=0.05,
            force_cpu=True
        ),
        MaxPool(Conv1d(100, 5, 0.8)),
        shared_parameters=True
    )
    embed_mapper = SequenceMapperSeq(
        VariationalDropoutLayer(train_config.var_dropout),
        recurrent_layer,
        VariationalDropoutLayer(train_config.var_dropout)
    )
    attention = BiAttention(TriLinear(bias=True), True)
    match_encoder = SequenceMapperSeq(
        FullyConnected(train_config.dim * 2, activation="relu"),
        ResidualLayer(SequenceMapperSeq(
            VariationalDropoutLayer(train_config.var_dropout),
            recurrent_layer,
            VariationalDropoutLayer(train_config.var_dropout),
            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
            FullyConnected(train_config.dim * 2, activation="relu"),
        )),
        VariationalDropoutLayer(train_config.var_dropout)
    )
    lm_model = LanguageModel(LM_VOCAB, LM_OPTIONS, LM_WEIGHTS, LM_TOKEN_WEIGHTS)
    model = CapeAttentionWithElmo(
        encoder=DocumentAndQuestionEncoder(answer_encoder),
        lm_model=lm_model,
        max_batch_size=train_config.max_batch_size,
        preprocess=preprocess,
        per_sentence=False,
        append_embed=(train_config.elmo_mode == "both" or train_config.elmo_mode == "input"),
        append_before_atten=(train_config.elmo_mode == "both" or train_config.elmo_mode == "output"),
        word_embed=word_embed,
        char_embed=char_embed,
        embed_mapper=embed_mapper,
        lm_reduce=None,
        lm_reduce_shared=lm_reduce,
        memory_builder=NullBiMapper(),
        attention=attention,
        match_encoder=match_encoder,
        predictor=predictor
    )
    return model


def prepare_data(model, train_config, dataset_oversampling, n_processes):
    extract = ExtractMultiParagraphsPerQuestion(
        MergeParagraphs(train_config.n_tokens),
        ShallowOpenWebRanker(train_config.num_paragraphs),
        model.preprocessor, intern=True
    )
    trivia_qa_test = RandomParagraphSetDatasetBuilder(
        train_config.test_batch_size,
        "merge" if train_config.trivia_qa_mode == "merge" else "group", True,
        train_config.oversample
    )
    trivia_qa_train = StratifyParagraphSetsBuilder(
        train_config.train_batch_size,
        train_config.trivia_qa_mode == "merge",
        True, train_config.oversample
    )

    datas = []
    for name, sampling in dataset_oversampling.items():
        for s in range(sampling):
            ds = TriviaQaSpanCorpus(name)
            ds.corpus_name = ds.corpus_name + '_{}'.format(s)
            datas.append(ds)

    data = MultiDataset(datas)
    data = PreprocessedData(data, extract, trivia_qa_train, trivia_qa_test, eval_on_verified=False)
    data.preprocess(n_processes, 1000)
    return data


def get_evaluators(train_config):
    return [
        LossEvaluator(),
        MultiParagraphSpanEvaluator(8, "triviaqa", train_config.trivia_qa_mode != "merge", per_doc=False)
    ]


def get_training_params(train_config):
    return TrainParams(
        SerializableOptimizer(
            train_config.optimizer,
            dict(learning_rate=train_config.learning_rate)
        ),
        num_epochs=train_config.n_epochs,
        ema=train_config.ema,
        max_checkpoints_to_keep=train_config.max_checkpoints_to_keep,
        async_encoding=train_config.async_encoding,
        log_period=train_config.log_period,
        eval_period=train_config.eval_period,
        save_period=train_config.save_period,
        best_weights=("dev", "b8/question-text-f1"),
        eval_samples=dict(dev=None, train=6000),
        eval_at_zero=False
    )


def run_training(savename: str,
                 train_config: TrainConfig,
                 dataset_oversampling: Dict[str, int],
                 n_processes: int,
                 use_cudnn: bool
                 ):
    """Train a Cape-Flavoured DocumentQA model.

    After preparing the datasets for training, a model will be created and saved in a directory
    specified by `savename`. Logging (Tensorboard) can be found in the log subdirectory of the model directory.

    The datasets to train the model on are specified in the `dataset_oversampling` dictionary.
    E.g. {'squad': 2, 'wiki':1} will train a model on one equivalence of triviaqa wiki and two equivalences of squad.

    :param savename: Name of model
    :param train_config: cape_config.TrainConfig object containing hyperparameters etc
    :param dataset_oversampling: dictionary mapping dataset names to integer counts of how much
       to oversample them
    :param n_processes: Number of processes to paralellize prepro on
    :param use_cudnn: Whether to train with GRU's optimized for Cudnn (recommended)
    """

    model = build_model(WithIndicators(), train_config, use_cudnn=use_cudnn)
    data = prepare_data(model, train_config, dataset_oversampling, n_processes)
    eval = get_evaluators(train_config)
    params = get_training_params(train_config)

    with open(__file__, "r", encoding='utf8') as f:
        notes = f.read()
    notes = "Mode: " + train_config.trivia_qa_mode + "\n" + notes
    notes += '\nDataset oversampling : ' + str(dataset_oversampling)

    # pull the trigger
    trainer.start_training(data, model, params, eval, model_dir.ModelDir(savename), notes)


def main():
    parser = argparse.ArgumentParser(description='Train a Cape-Flavoured DocumentQA model.')
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument('-s', '--dataset_sampling', dest='dataset_sampling'
                        , default='', help='path to sampling_dict.json (see readme for more info')
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with"
                        )
    parser.add_argument('--cudnn', action='store_true', dest='cudnn', help='')
    parser.add_argument('--no_cudnn', action='store_false', dest='cudnn', help='')
    parser.set_defaults(cudnn=True)
    args = parser.parse_args()
    if args.dataset_sampling == '':
        dataset_sampling = {'wiki': 1, 'web': 1, 'squad': 1}
    else:
        dataset_sampling = json.load(open(args.dataset_sampling))
    train_conf = TrainConfig()
    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")
    run_training(out, train_conf, dataset_sampling, args.n_processes, args.cudnn)


if __name__ == "__main__":
    main()