from docqa import trainer
import horovod.tensorflow as hvd
import time
from os.path import join, relpath
from typing import List, Union
import numpy as np
import tensorflow as tf
from docqa.dataset import TrainingData
from docqa.evaluator import Evaluator, AysncEvaluatorRunner
from docqa.model import Model
from docqa.model_dir import ModelDir
from threading import Thread


def _build_train_ops(train_params):
    """ Bulid ops we should run during training, including learning, EMA, and summary ops"""
    global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                  initializer=tf.constant_initializer(0), trainable=False)
    #global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection(tf.GraphKeys.LOSSES)
    if len(loss) == 0:
        raise RuntimeError("No losses found in losses collection")
    loss = tf.add_n(loss, name="loss")

    if len(tf.get_collection(tf.GraphKeys.SUMMARIES)) > 0:
        # Add any summaries client stored in SUMMARIES
        summary_tensor = tf.summary.merge([[tf.summary.tensor_summary("loss", loss)] +
                                           tf.get_collection(tf.GraphKeys.SUMMARIES)])
    else:
        summary_tensor = tf.summary.tensor_summary("loss", loss)

    train_objective = loss

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(regularizers) > 0:
        regularization_loss = tf.add_n(regularizers, name="regularization_loss")
        if train_params.regularization_weight is not None:
            train_objective = train_objective + regularization_loss * train_params.regularization_weight
        else:
            train_objective = train_objective + regularization_loss
    else:
        regularization_loss = None

    opt = train_params.opt.get()
    opt = hvd.DistributedOptimizer(opt)
    #train_opt = opt.apply_gradients(opt.compute_gradients(train_objective), global_step=global_step)
    train_opt = opt.minimize(train_objective, global_step=global_step)

    if train_params.ema is not None:
        ema = tf.train.ExponentialMovingAverage(decay=train_params.ema)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_opt]):
            # Run the old training op, then update the averages.
            train_opt = tf.group(ema_op)
    else:
        ema = None

    # Any collections starting with "monitor" are also added as summaries
    to_monitor = {}
    for col in tf.get_default_graph().get_all_collection_keys():
        if col.startswith("monitor"):
            v = tf.get_collection(col)
            if len(v) > 0:
                print("Monitoring: " + col)
                v = tf.add_n(v)
                to_monitor[col] = v

    if len(to_monitor) > 0:
        monitor_ema = tf.train.ExponentialMovingAverage(decay=train_params.monitor_ema, name="MonitorEMA",
                                                        zero_debias=True)
        train_opt = tf.group(train_opt, monitor_ema.apply(list(to_monitor.values())))
        summary_tensor = tf.summary.merge(
            [tf.summary.scalar(col, monitor_ema.average(v)) for col, v in to_monitor.items()] +
            [summary_tensor])

    # EMA for the loss and what we monitoring
    if train_params.loss_ema is not None:
        loss_ema = tf.train.ExponentialMovingAverage(decay=train_params.loss_ema, name="LossEMA", zero_debias=True)

        if regularization_loss is None:
            ema_op = loss_ema.apply([loss])
            train_opt = tf.group(train_opt, ema_op)
            ema_var = loss_ema.average(loss)
            summary_tensor = tf.summary.merge([tf.summary.scalar("training-ema/loss", ema_var), summary_tensor])
        else:
            to_track = [loss, train_objective, regularization_loss]
            ema_op = loss_ema.apply(to_track)
            train_opt = tf.group(train_opt, ema_op)
            tensor_vars = [
                tf.summary.scalar("training-ema/loss", loss_ema.average(loss)),
                tf.summary.scalar("training-ema/objective", loss_ema.average(train_objective)),
                tf.summary.scalar("training-ema/regularization-loss",
                                  loss_ema.average(regularization_loss))
                ]
            summary_tensor = tf.summary.merge([tensor_vars, summary_tensor])

    return loss, summary_tensor, train_opt, global_step, ema


def _train(model: Model,
           data: TrainingData,
           checkpoint: Union[str, None],
           parameter_checkpoint: Union[str, None],
           save_start: bool,
           train_params: trainer.TrainParams,
           evaluators: List[Evaluator],
           out: ModelDir,
           notes=None,
           dry_run=False,
           start_eval=False):
    print('Horovod size: ', hvd.size())
    print('Horovod rank: ', hvd.rank())
    print('Horovod local rank: ', hvd.local_rank())

    if train_params.async_encoding:
        _train_async(model, data, checkpoint, parameter_checkpoint, save_start, train_params,
                 evaluators, out, notes, dry_run, start_eval)
        return
    else:
        raise NotImplementedError('Syncronous training with Horovod not supported yet')


def _train_async(model: Model,
                 data: TrainingData,
                 checkpoint: Union[str, None],
                 parameter_checkpoint: Union[str, None],
                 save_start: bool,
                 train_params: trainer.TrainParams,
                 evaluators: List[Evaluator],
                 out: ModelDir,
                 notes=None,
                 dry_run=False,
                 start_eval=False):
    """ Train while encoding batches on a seperate thread and storing them in a tensorflow Queue, can
    be much faster then using the feed_dict approach """

    train = data.get_train()

    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()

    print("Training on %d batches" % len(train))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, len(data)) for name, data in eval_datasets.items()))

    # spec the model for the given datasets
    model.set_inputs([train] + list(eval_datasets.values()), loader)
    placeholders = model.get_placeholders()

    train_queue = tf.FIFOQueue(train_params.async_encoding, [x.dtype for x in placeholders], name="train_queue")
    evaluator_runner = AysncEvaluatorRunner(evaluators, model, train_params.async_encoding)
    train_enqeue = train_queue.enqueue(placeholders)
    train_close = train_queue.close(True)

    is_train = tf.placeholder(tf.bool, ())
    input_tensors = tf.cond(is_train, lambda: train_queue.dequeue(),
                            lambda: evaluator_runner.eval_queue.dequeue())

    # tensorfow can't infer the shape for an unsized queue, so set it manually
    for input_tensor, pl in zip(input_tensors, placeholders):
        input_tensor.set_shape(pl.shape)

    bcast = hvd.broadcast_global_variables(0)
    print("Init model...")
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    with sess.as_default():
        pred = model.get_predictions_for(dict(zip(placeholders, input_tensors)))

    evaluator_runner.set_input(pred)

    if parameter_checkpoint is not None:
        print("Restoring parameters from %s" % parameter_checkpoint)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        saver = None

    print("Setting up model prediction / tf...")
    all_vars = tf.global_variables()

    loss, summary_tensor, train_opt, global_step, weight_ema = _build_train_ops(train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    if train_params.best_weights is not None:
        lst = all_vars
        if weight_ema is not None:
            for x in lst:
                v = weight_ema.average(x)
                if v is not None:
                    lst.append(v)
        best_weight_saver = tf.train.Saver(var_list=lst, max_to_keep=1)
        cur_best = None
    else:
        best_weight_saver = None
        cur_best = None

    saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
    summary_writer = tf.summary.FileWriter(out.log_dir)

    # Load or initialize the model parameters
    if checkpoint is not None:
        print("Restoring from checkpoint...")
        saver.restore(sess, checkpoint)
        print("Loaded checkpoint: " + str(sess.run(global_step)))
    else:
        print("Initializing parameters...")
        sess.run(tf.global_variables_initializer())
    sess.run(bcast)

    # Make sure no bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
    tf.get_default_graph().finalize()

    if dry_run:
        return

    on_step = sess.run(global_step)

    if save_start:
        # summary_writer.add_graph(sess.graph, global_step=on_step)
        if hvd.rank() == 0:
            trainer.save_train_start(out.dir, data, sess.run(global_step), evaluators, train_params, notes)

    def enqueue_train():
        try:
            # feed data from the dataset iterator -> encoder -> queue
            for epoch in range(train_params.num_epochs):
                for batch in train.get_epoch():
                    feed_dict = model.encode(batch, True)
                    sess.run(train_enqeue, feed_dict)
        except tf.errors.CancelledError:
            # The queue_close operator has been called, exit gracefully
            return
        except Exception as e:
            # Crashes the main thread with a queue exception
            sess.run(train_close)
            raise e

    train_enqueue_thread = Thread(target=enqueue_train)
    train_enqueue_thread.daemon = True  # Ensure we exit the program on an excpetion

    print("Start training!")

    batch_time = 0

    train_dict = {is_train: True}
    eval_dict = {is_train: False}
    try:
        train_enqueue_thread.start()

        for epoch in range(train_params.num_epochs):
            for batch_ix in range(len(train)):
                t0 = time.perf_counter()
                on_step = sess.run(global_step) + 1
                get_summary = on_step % train_params.log_period == 0

                if get_summary:
                    summary, _, batch_loss = sess.run([summary_tensor, train_opt, loss], feed_dict=train_dict)
                else:
                    summary = None
                    _, batch_loss = sess.run([train_opt, loss], feed_dict=train_dict)

                if np.isnan(batch_loss):
                    raise RuntimeError("NaN loss!")

                batch_time += time.perf_counter() - t0
                if hvd.rank() == 0:
                    if summary is not None:
                        print("on epoch=%d batch=%d step=%d, time=%.3f" %
                              (epoch, batch_ix + 1, on_step, batch_time))
                        summary_writer.add_summary(
                            tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]), on_step)
                        summary_writer.add_summary(summary, on_step)
                        batch_time = 0

                # occasional saving
                if hvd.rank() == 0:
                    if on_step % train_params.save_period == 0:
                        print("Checkpointing")
                        saver.save(sess, join(out.save_dir, "checkpoint-" + str(on_step)), global_step=global_step)
                # Occasional evaluation
                if (on_step % train_params.eval_period == 0) or start_eval:
                    print("Running evaluation...")
                    start_eval = False
                    t0 = time.perf_counter()
                    for name, data in eval_datasets.items():
                        n_samples = train_params.eval_samples.get(name)
                        evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples, eval_dict)
                        if hvd.rank() == 0:
                            for s in evaluation.to_summaries(name + "-"):
                                summary_writer.add_summary(s, on_step)

                            # Maybe save as the best weights
                            if train_params.best_weights is not None and name == train_params.best_weights[0]:
                                val = evaluation.scalars[train_params.best_weights[1]]
                                if cur_best is None or val > cur_best:
                                    print("Save weights with current best weights (%s vs %.5f)" % (
                                        "None" if cur_best is None else ("%.5f" % cur_best), val))
                                    best_weight_saver.save(sess, join(out.best_weight_dir, "best"), global_step=global_step)
                                    cur_best = val

                            print("Evaluation took: %.3f seconds" % (time.perf_counter() - t0))
    finally:
        sess.run(train_close)  # terminates the enqueue thread with an exception

    train_enqueue_thread.join()

    saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)
    sess.close()


trainer._build_train_ops = _build_train_ops
trainer._train = _train
trainer._train_async = _train_async
