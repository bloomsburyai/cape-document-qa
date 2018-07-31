from cape_document_qa import patches
from os import mkdir, listdir
from shutil import copyfile
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, ParagraphAndQuestion
from docqa.utils import ResourceLoader
import argparse
from docqa.nn.recurrent_layers import BiRecurrentMapper, CompatGruCellSpec
import pickle
from os.path import exists, join, isfile
import numpy as np
import tensorflow as tf
from docqa.model_dir import ModelDir


def get_test_questions():
    paragraph = ["Harry", "Potter", "was", "written", "by", "JK"]
    question = ["Who", "wrote", "Harry", "Potter", "?"]
    return ParagraphAndQuestion(paragraph, question, None, "test_questions")


def get_map_indim():
    # ToDo: make this less arbitrary
    return 400 + 1024


def get_dims(x, dim):
    if "map_embed" in x.name:
        outdim = dim
        indim = get_map_indim()
    elif "chained-out" in x.name:
        outdim = dim
        indim = dim * 4
    else:
        outdim = dim
        indim = dim * 2
    return indim, outdim


def convert_saved_graph(model_dir, output_dir):
    print("Load model")
    md = ModelDir(model_dir)
    model = md.get_model()

    # remove the lm models word embeddings - cpu model will use Char-CNN
    model.lm_model.embed_weights_file = None
    dim = model.embed_mapper.layers[1].n_units

    print("Setting up cudnn version")
    sess = tf.Session()
    with sess.as_default():
        model.set_input_spec(
            ParagraphAndQuestionSpec(1, None, None, 14),
            {"the"}, ResourceLoader(lambda a, b: {"the": np.zeros(300, np.float32)})
        )
        print("Buiding graph")
        pred = model.get_prediction()

    test_questions = get_test_questions()

    print("Load vars:")
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars = [x for x in all_vars if x.name not in lm_var_names]
    md.restore_checkpoint(sess, vars)
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))

    feed = model.encode([test_questions], False)
    cuddn_out = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)

    print("Done, copying files...")
    if not exists(output_dir):
        mkdir(output_dir)
    for file in listdir(model_dir):
        if isfile(file) and file != "model.npy":
            copyfile(join(model_dir, file), join(output_dir, file))

    print("Done, mapping tensors...")
    to_save, to_init = [], []
    for x in tf.trainable_variables():
        if x.name.endswith("/gru_parameters:0"):
            key = x.name[:-len("/gru_parameters:0")]
            indim, outdim = get_dims(x, dim)
            c = cudnn_rnn_ops.CudnnGRUSaveable(x, 1, outdim, indim, scope=key)
            for spec in c.specs:
                if spec.name.endswith("bias_cudnn 0") or \
                        spec.name.endswith("bias_cudnn 1"):
                    print('Unsupported spec: ' + spec.name)
                    continue
                if 'forward' in spec.name:
                    new_name = spec.name.replace('forward/rnn/multi_rnn_cell/cell_0/', 'bidirectional_rnn/fw/')
                else:
                    new_name = spec.name.replace('backward/rnn/multi_rnn_cell/cell_0/', 'bidirectional_rnn/bw/')
                v = tf.Variable(sess.run(spec.tensor), name=new_name)
                to_init.append(v)
                to_save.append(v)
        else:
            to_save.append(x)

    save_dir = join(output_dir, "save")
    if not exists(save_dir):
        mkdir(save_dir)

    # save:
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    vars_to_save = [x for x in all_vars if not x.name.startswith("bilm")]
    sess.run(tf.initialize_variables(to_init))
    saver = tf.train.Saver(vars_to_save)
    saver.save(
        sess,
        join(save_dir, 'checkpoint'),
        global_step=123456789,
        write_meta_graph=False,
    )

    sess.close()
    tf.reset_default_graph()
    return cuddn_out


def convert_model_pickle(model_dir, output_dir):
    print("Updating model...")
    md = ModelDir(model_dir)
    model = md.get_model()
    # remove the lm models word embeddings - cpu model will use Char-CNN
    model.lm_model.embed_weights_file = None
    dim = model.embed_mapper.layers[1].n_units

    model.embed_mapper.layers = [
        model.embed_mapper.layers[0],
        BiRecurrentMapper(CompatGruCellSpec(dim)), model.embed_mapper.layers[2]
    ]
    model.match_encoder.layers = list(model.match_encoder.layers)
    other = model.match_encoder.layers[1].other
    other.layers = list(other.layers)
    other.layers[1] = BiRecurrentMapper(CompatGruCellSpec(dim))

    pred = model.predictor.predictor
    pred.first_layer = BiRecurrentMapper(CompatGruCellSpec(dim))
    pred.second_layer = BiRecurrentMapper(CompatGruCellSpec(dim))

    with open(join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def test_model_pickle(output_dir):
    print("Testing...")
    save_dir = join(output_dir, "save")
    test_questions = get_test_questions()

    with open(join(output_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    sess = tf.Session()
    model.set_input_spec(ParagraphAndQuestionSpec(1, None, None, 14),
                         {"the"},
                         ResourceLoader(lambda a, b: {"the": np.zeros(300, np.float32)}))
    pred = model.get_prediction()

    print("Rebuilding")
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars_to_restore = [x for x in all_vars if x.name not in lm_var_names]
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))

    feed = model.encode([test_questions], False)
    cpu_out = sess.run([pred.start_logits, pred.end_logits], feed_dict=feed)
    return cpu_out


def convert(model_dir, output_dir):
    if exists(output_dir):
        raise Exception('Output destination already exists')
    gpu_out = convert_saved_graph(model_dir, output_dir)
    convert_model_pickle(model_dir, output_dir)
    cpu_out = test_model_pickle(output_dir)
    print('CPU Logits:')
    print(cpu_out)
    print('GPU Logits:')
    print(gpu_out)
    print('CPU SOFTMAXED LOGITS:')
    print(np.exp(cpu_out) / np.sum(np.exp(cpu_out)))
    print('GPU SOFTMAXED LOGITS:')
    print(np.exp(gpu_out) / np.sum(np.exp(gpu_out)))
    success = all([np.allclose(a, b) for a, b in zip(cpu_out, gpu_out)])
    if success:
        print('Conversion Successful')
    else:
        print('Conversion unsuccessful, logits dont match closely')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    convert(args.target_model, args.output_dir)


if __name__ == "__main__":
    main()
