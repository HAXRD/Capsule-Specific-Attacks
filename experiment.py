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

tf.flags.DEFINE_string('mode', 'train',
                       'train, test, visual, dream')

tf.flags.DEFINE_integer('hparams_override', None,
                        'A string of form key=value,key=value to override the'
                        'hparams of this experiment.')
tf.flags.DEFINE_string('data_dir', './data/cifar-10-batches-bin', 
                       'The data directory.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'The dataset to use for the experiment.'
                       'mnist, norb, cifar10.')
tf.flags.DEFINE_string('model', 'cnn',
                       'The model to use for the experiment.'
                       'capsule or cnn')
tf.flags.DEFINE_integer('total_batch_size', 100, 'Total batch size.')
tf.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
tf.flags.DEFINE_string('summary_dir', './summary',
                       'Main directory for the experiments.')
tf.flags.DEFINE_integer('save_epochs', 5, 'How often to save checkpoints.')
tf.flags.DEFINE_integer('max_epochs', 1500, 'Number of epochs to train.')

models = {
    'cnn': cnn_model.CNNModel
}

def get_distributed_batched_dataset(total_batch_size, num_gpus, 
                             data_dir, dataset, split='default'):
    """Reads the input data and set the batch_size to 1/`num_gpus`
    of the `total_batch_size`. 

    Args:
        total_batch_size: Total number of data entries over all towers.
        num_gpus: The number of GPUs available to use.
        data_dir: The directory containing the data.
        dataset: The name of dataset.
        split: 'train', 'test', 'default' (only for noise data)
    Returns:
        bat: A list of batched dataset
    """
    # Distribute the total batch into `num_gpus` partitions
    batch_size = total_batch_size // max(1, num_gpus)
    with tf.device('/gpu:0'):
        if dataset == 'mnist':
            raise NotImplementedError('mnist not implemented yet!')
        elif dataset == 'norb':
            raise NotImplementedError('norb not implemented yet!')
        elif dataset == 'cifar10':
            distributed_batched_dataset, specs = cifar10_input.inputs(
                split, data_dir, batch_size)
        else:
            raise ValueError(
                'Unexpected dataset {} read!'.format(dataset))
    return distributed_batched_dataset, specs

def extract_step(path):
    """Returns the step from the file format name of Tensorflow checkpoints.

    Args:
        path: The checkpoint path returned by tf.train.get_checkpoint_state.
        The format is: {ckpnt_name}-{step}
    Returns:
        The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])

def load_training(saver, sess, load_dir):
    """Loads a saved model into current session or initializes the directory.

    If there is no functioning saved model or FLAGS.restart is set, cleans the
    load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
    to session.

    Args:
        saver: An instance of tf.train.saver to load the model in to the session.
        session: An instance of tf.Session with the built-in model graph.
        load_dir: The directory which is used to load the latest checkpoint.
    Returns:
        The latest saved step.
    """
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)
        else:
            tf.gfile.DeleteRecursively(load_dir) 
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step

def run_train_session(iterator, specs, num_gpus, # Dataset related
                      latest_epoch_loader, summary_dir, # Checkpoint related
                      result, save_epochs, max_epochs): # Model related
    """Starts a session, 

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus
        latest_epoch_loader: func, the function that load the latest
            checkpoint.
        summary_dir: str, directory storing the checkpoints.
        result: namedtuple, JoinedResult('summary', 'train_op', 'correct')
        save_epochs: scalar, how often to save the model
        max_epochs: scalar, maximum number of epochs
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Declare summary writer and save the graph in the meanwhile.
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        # Declare batched data instance and initialize the iterator
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)
        # Initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # Declare saver object for future saving
        saver = tf.train.Saver(max_to_keep=1000)
        # Load the latest step if any
        latest_epoch = latest_epoch_loader(saver, sess, summary_dir)

        # Start feeding process
        for i in range(latest_epoch, max_epochs): # epoch loop
            step_counter = (i + 1) * specs['steps_per_epoch']
            # Initialize dataset
            sess.run(iterator.initializer)
            while True:
                try:
                    step_counter += 1
                    # Declare a list to store the batched data
                    batch_vals = []
                    for j in range(num_gpus): # gpu loop
                        batch_vals.append(sess.run(batch_data))
                    
                    # Get placeholders and create feed_dict
                    feed_dict = {}
                    placeholders = tf.get_collection('placeholders')
                    for j, batch_val in enumerate(batch_vals):
                        for ph in placeholders:
                            if 'tower_%d' % j in ph.name:
                                if 'batched_images' in ph.name:
                                    feed_dict[ph] = batch_val['images']
                                elif 'batched_labels' in ph.name:
                                    feed_dict[ph] = batch_val['labels']

                    summary, accuracy, _ = sess.run(
                        [result.summary, result.accuracy, result.train_op], 
                        feed_dict=feed_dict)
                    writer.add_summary(summary, step_counter)
                except tf.errors.OutOfRangeError:
                    break
                # Finished one step
            if (i + 1) % save_epochs == 0:
                global_step = (i + 1) * specs['steps_per_epoch']
                ckpt_path = saver.save(
                    sess, os.path.join(summary_dir, 'model.ckpt'), 
                    global_step=global_step)
                tf.logging.info("{} epochs done (step = {}). Accuracy {}. Checkpoint {}".format(
                    i+1, global_step, accuracy, ckpt_path))

        """Debug

            for i in range(max_epochs): # epochs loop
                # Intialize iterator for each epoch
                sess.run(iterator.initializer) 
                # For each epoch, we can at most run the loop 
                # 50000/100=500 times
                counter = 0
                while True:
                    try:
                        for j in range(num_gpus): # gpu loop
                            # Get single batch data (a dictionary)
                            batch_val = sess.run(batch_data)
                            print(batch_val['images'].shape)
                            print(batch_val['labels'].shape)
                        counter += 1
                    except tf.errors.OutOfRangeError:
                        break
                print(counter) # ==> 500
        """

def train(hparams, data_dir, dataset, model_type, total_batch_size,
                   num_gpus, summary_dir, 
                   save_epochs, max_epochs):
    """Trains a model with batch sizes of 100 to 50000/100*`max_epochs` steps.

    It will initialize the model with either previously saved model in the 
    `summary_dir` or start from scratch if the directory is empty.
    The training is distributed on `num_gpus` GPUs. It writes a summary at 
    every step and saves the model every `save_epochs` steps.

    Args:
        hparams: The hyperparameters to build the model graph.
        data_dir: The directory containing the input data.
        dataset: The name of the dataset for the experiments.
        model_type: The name of the model architecture.
        total_batch_size: Total batch size, will be distributed to `num_gpus` GPUs.
        num_gpus: The number of GPUs available.
        summary_dir: The directory to write summaries and save the model.
        save_epochs: How often the training model should be saved.
        max_epochs: Maximum epochs to train.
    """
    summary_dir += '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default() as graph:
        """ Simple example to load batched input

            # total size for test = 10,000
            batched_dataset = cifar10_input.inputs('test', data_dir, 5000) 
            iterator = batched_dataset.make_initializable_iterator()
            batch_data = iterator.get_next()
            with tf.Session() as sess:
                sess.run(iterator.initializer)
                while True:
                    try:
                        batch_val = sess.run(batch_data)
                        print(batch_val['images'].shape) # ==> (5000, 3, 32, 32) x 2
                        print(batch_val['labels'].shape) # ==> (5000, 10) x 2
                    except tf.errors.OutOfRangeError:
                        break
        """
        # Get batched dataset and declare initializable iterator
        distributed_batched_dataset, dataset_specs = get_distributed_batched_dataset(
            total_batch_size, num_gpus, data_dir, dataset, 'train')
        iterator = distributed_batched_dataset.make_initializable_iterator()
        # Initialize model with hparams and dataset_specs
        model = models[model_type](hparams, dataset_specs)
        # Build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        result, _ = model.build_model_on_multi_gpus(num_gpus)
        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        # Call train experiment
        run_train_session(iterator, dataset_specs, num_gpus,
                          load_training, summary_dir,
                          result, save_epochs, max_epochs)

def default_hparams():
    """Builds an HParams object with default hperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=2000,
        leaky=False,
        learning_rate=0.001,
        loss_type='margin',
        num_prime_capsules=32,
        padding='VALID',
        remake=True,
        routing=3,
        verbose=False)

def main(_):
    hparams = default_hparams()
    if FLAGS.hparams_override:
        hparams.parse(FLAGS.hparams_override)

    if FLAGS.mode == 'train':
        train(hparams, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size,
                       FLAGS.num_gpus, FLAGS.summary_dir, 
                       FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'test':
        pass
    elif FLAGS.mode == 'visual':
        pass
    elif FLAGS.mode == 'dream':
        pass
    else:
        raise NotImplementedError(
            "No matching mode found for '{}'".format(FLAGS.mode))
    


if __name__ == '__main__':
    tf.app.run()