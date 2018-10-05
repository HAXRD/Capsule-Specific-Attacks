# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Framework for training and evaluating models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np 
import tensorflow as tf 

from input_data.cifar10 import cifar10_input

from models import cnn_model

FLAGS = tf.flags.FLAGS

# share
tf.flags.DEFINE_string('data_dir', None, 'The data directory.')
tf.flags.DEFINE_string('hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'hparams of this experiment.')
tf.flags.DEFINE_string('model', None,
                       'The model to use for the experiment'
                       'capsule or cnn')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'the dataset to use for the experiment.'
                       'mnist, norb, cifar10')
tf.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
tf.flags.DEFINE_integer('num_targets', 1,
                        'Number of targets to detect (1 or 2).')
tf.flags.DEFINE_string('checkpoint', None,
                       'The model checkpoint for evaluation.')
tf.flags.DEFINE_string('mode', None,
                       'train or test or dream')

# train
tf.flags.DEFINE_integer('max_steps', 1000, 'Number of steps to train.')
tf.flags.DEFINE_integer('save_step', 1500, 'How often to save ckpt files.')
tf.flags.DEFINE_string('summary_dir', None,
                       'Summary directory for the experiments.')

# test
tf.flags.DEFINE_integer('eval_size', 10000, 'Size of the test dataset.')
tf.flags.DEFINE_integer('num_trials', 1,
                        'Number of trials for ensemble evaluation.')

# dream
# TODO:

models = {
    'cnn_model': cnn_model.CNNModel
}

def train_experiment(session, result, writer, last_step, max_steps, saver, 
                     summary_dir, save_step):
    """Runs training for up to max_steps and saves the model and summaries.

    Args:
        session: The loaded tf.session with the initialized model.
        result: The resultant operations of the model including train_op.
        writer: The summary writer file.
        last_step: The last trained step.
        max_steps: Maximum nnumber of training iterations.
        saver: An instance of tf.train.saver to save the current model.
        summary_dir: The directory to save the model in it.
        save_step: How often to save the model ckpt.
    """
    step = 0
    for i in range(last_step, max_steps):
        step += 1
        summary, _ = session.run([result.summary, result.train_op]) # summary and apply_gradient op
        writer.add_summary(summary, i)
        if (i + 1) % save_step == 0:
            saver.save(
                session, os.path.join(summary_dir, 'model.ckpt'), global_step = i + 1)

def extract_step(path):
    """Returns the step from the file format name of Tensorflow checkpoints

    Args:
        path: The checkpoint path returned by tf.train.get_checkpoint_state.
            The format is: {ckpt_name}-{step}
    Returns:
        The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])

def load_training(saver, session, load_dir):
    """Loads a saved model into current session or initializes the directory.

    If there is no functioning saved mdoel or FLAGS.restart is set, cleans the 
    load_dir directory. Othewise, loads the lastest saved checkpoint in load_dir
    to session.

    Args:
        saver: An instance of tf.train.saver to load the model into the session.
        session: An instance of tf.Session with the build-in model graph.
        load_dir: The directory which is used to load the latest checkpoint.
    Returns:
        The lastest saved step.
    """
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)
        else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step

def run_experiment(loader, load_dir, writer, experiment, result, max_steps, save_step=0):
    """Starts a session, loads the model and runs the given experiment on it.

    This is a general wrapper to load a saved model and run an experiment on it.
    An experiment can be a training experiment or an evaluation experiment or a dreaming
    experiment. It starts session, threads and queues and closes them before returning.

    Args:
        loader: A function of prototype (saver, session, load_dir) to load a saved
            checkpoint in load_dir given a session and saver.
        load_dir: The directory to load the previously saved model from it and to 
            save the current model in it.
        writer: A tf.summary.FileWriter to add summarie.
        experiment: The function of prototype (session, result, writer, last_step,
            max_steps, saver, load_dir, save_step) which will execute the experiment
            steps from result on the given session.
        result: The resultant final operations of the built model.
        max_steps: Maximum number of experiment iterations.
        save_step: How often the training model should be saved.
    """
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    session.run(init_op)
    saver = tf.train.Saver(max_to_keep=1000)
    last_step = loader(saver, session, load_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    try:
        experiment(
            session=session,
            result=result,
            writer=writer,
            last_step=last_step,
            max_steps=max_steps,
            saver=saver,
            summary_dir=load_dir,
            save_step=save_step)
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished experiment.')
    finally:
        coord.request_stop()
    coord.join(threads)
    session.close()

def get_features(split, total_batch_size, num_gpus, data_dir, num_targets,
                 dataset):
    """Reads the input data and distributes it over num_gpus GPUs.

    Each tower of data has 1/FLAGS.num_gpus of the total_batch_size

    Args:
        split: 'train' or 'test' or 'dream', split of the data to read.
        total_batch_size: total number of data entries over all towers.
        num_gpus: Number of GPUs to distribute the data on.
        data_dir: Directory containing the input data.
        num_targets: Number of objects present in the image.
        dataset: The name of the dataset, 'mnist', 'norb' or 'cifar10'
    Returns:
        A list of batched feature dictionaries.
    Raises:
        ValueError: If dataset is not 'mnist', 'norb' or 'cifar10'
        NotImplemented: 
    """
    batch_size = total_batch_size // max(1, num_gpus)
    features = []
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            if dataset == 'mnist':
                raise NotImplementedError('mnist not implemented!')
                # TODO:implement mnist
            elif dataset == 'norb':
                raise NotImplementedError('norb not implemented!')
            elif dataset == 'cifar10':
                features.append(cifar10_input.inputs(
                    split=split, data_dir=data_dir, batch_size=batch_size))
            else:
                raise ValueError(
                    'Unexpected dataset {!r}, must be mnist, norb, or cifar10'.format(
                        dataset))
    return features

def train(hparams, summary_dir, num_gpus, model_type, max_steps, save_step,
          data_dir, num_targets, dataset):
    """Trains a model with batch sizes of 128 to FLAGS.max_steps steps.
    
    It will initialize the model with either previously saved model in the
    summary directory or start from scratch if FLAGS.restart is set or the
    directory is empty.
    The training is distributed on num_gpus GPUs. It writes a summary at 
    every step and saves the model every 1500 iterations.

    Args:
        hparams: The hyperparameters to build the model graph.
        summary_dir: The directory to save model and write training summaries.
        num_gpus: Number of GPUs to use for reading data and computation.
        model_type: The model architecture category
        max_steps: Maximum number of training iterations
        save_step: How often the training model should be saved.
        data_dir: Directory containing the input data.
        num_targets: Number of objects present in the image.
        dataset: Name of the dataset for the experiment.
    """
    summary_dir = '/train/'
    with tf.Graph().as_default():
        # Build model
        features = get_features(
            'train', 128, num_gpus, data_dir, num_targets, dataset)
        model = models[model_type](hparams)
        result, _ = model.multi_gpu(features, num_gpus)
        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        writer = tf.summary.FileWriter(summary_dir)
        run_experiment(load_training, summary_dir, writer, train_experiment, result,
                       max_steps, save_step)
        writer.close()


def default_hparams():
    """Builds an HParams object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=2000,
        leaky=False,
        learning_rate=0.001,
        loss_type='margin',
        num_prime_capsules=64,
        padding='VALID',
        remake=True,
        routing=3,
        verbose=False
    )

def main(_):
    hparams = default_hparams()
    if FLAGS.hparams_override:
        hparams.parse(FLAGS.hparams_override)
    
    if FLAGS.mode == 'train':
        train(hparams, FLAGS.summary_dir, FLAGS.num_gpus, FLAGS.model,
              FLAGS.max_steps, FLAGS.save_step, FLAGS.data_dir, FLAGS.num_targets,
              FLAGS.dataset)
    elif FLAGS.mode == 'test':
        # TODO
        pass
    elif FLAGS.mode == 'dream':
        # TODO
        pass

if __name__ == '__main__':
    tf.app.run()