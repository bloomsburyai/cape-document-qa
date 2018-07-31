import tensorflow as tf
from docqa import trainer
import horovod.tensorflow as hvd
import time
from os.path import exists, join, relpath
from typing import List, Union, Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from docqa.dataset import TrainingData
from docqa.evaluator import Evaluator, EvaluatorRunner
from docqa.model import Model
from docqa.model_dir import ModelDir


def _build_train_ops(train_params):
    """ Bulid ops we should run during training, including learning, EMA, and summary ops"""
    global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                  initializer=tf.constant_initializer(0), trainable=False)

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
    train_opt = opt.minimize(train_objective)

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
    if train_params.async_encoding:
        raise Exception()

    if train_params.best_weights is not None:
        raise NotImplementedError

    # spec the model for the current voc/input/batching
    train = data.get_train()
    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()
    evaluator_runner = EvaluatorRunner(evaluators, model)

    print("Training on %d batches" % len(train))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, len(data)) for name, data in eval_datasets.items()))

    print("Init model...")
    model.set_inputs([train] + list(eval_datasets.values()), loader)

    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    print("Setting up model prediction / tf...")

    loss, summary_tensor, train_opt, global_step, _ = _build_train_ops(train_params)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=out.save_dir,
                                           config=config,
                                           hooks=hooks) as sess:

        with sess.as_default():
            pred = model.get_prediction()
        evaluator_runner.set_input(pred)

        # Pre-compute tensors we need at evaluations time
        eval_tensors = []
        for ev in evaluators:
            eval_tensors.append(ev.tensors_needed(pred))
        if hvd.rank() == 0:
            saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
        summary_writer = tf.summary.FileWriter(out.log_dir)

        # Load or initialize the model parameters
        if parameter_checkpoint is not None:
            print("Initializing training variables...")
            vars = [x for x in tf.global_variables() if x not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            sess.run(tf.variables_initializer(vars))
        else:
            print("Initializing parameters...")
            sess.run(tf.global_variables_initializer())

        # Make sure no bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
        tf.get_default_graph().finalize()

        print("Start training!")

        on_step = sess.run(global_step)
        if save_start:
            summary_writer.add_graph(sess.graph, global_step=on_step)
            trainer.save_train_start(out.dir, data, on_step, evaluators, train_params, notes)
        if hvd.rank() == 0:
            saver.save(sess, join(out.save_dir, "checkpoint-" + str(on_step)), global_step=global_step)

        batch_time = 0
        for epoch in range(train_params.num_epochs):
            for batch_ix, batch in enumerate(tqdm(train.get_epoch())):
                t0 = time.perf_counter()
                on_step = sess.run(global_step) + 1  # +1 because all calculations are done after step

                get_summary = on_step % train_params.log_period == 0
                encoded = model.encode(batch, True)

                if get_summary:
                    summary, _, batch_loss = sess.run([summary_tensor, train_opt, loss], feed_dict=encoded)
                else:
                    summary = None
                    _, batch_loss = sess.run([train_opt, loss], feed_dict=encoded)

                if np.isnan(batch_loss):
                    raise RuntimeError("NaN loss!")

                batch_time += time.perf_counter() - t0
                if hvd.rank() == 0:

                    if get_summary:

                        print("on epoch=%d batch=%d step=%d time=%.3f" %
                              (epoch, batch_ix + 1, on_step, batch_time))

                        summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]),
                                                   on_step)
                        summary_writer.add_summary(summary, on_step)
                        batch_time = 0

                    # occasional saving
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
                        evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples)
                        for s in evaluation.to_summaries(name + "-"):
                            summary_writer.add_summary(s, on_step)

                    print("Evaluation took: %.3f seconds" % (time.perf_counter() - t0))
        if hvd.rank() == 0:
            saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)


trainer._build_train_ops = _build_train_ops
trainer._train = _train
