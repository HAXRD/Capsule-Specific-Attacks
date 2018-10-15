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
from input_data.noise import noise_input_
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
tf.flags.DEFINE_integer('max_to_keep', None, 
                        'Maximum number of checkpoint files to keep.')
tf.flags.DEFINE_integer('save_epochs', 5, 'How often to save checkpoints.')
tf.flags.DEFINE_integer('max_epochs', 1500, 'Number of epochs to train.')
tf.flags.DEFINE_integer('n_repeats', 10, 
                        'How many repeats of noise dataset to create in'
                        'order to visualize the available layers (hard coded).'
                        'cnn 1806 = 512+1 + 256+1 + 1024+1 + 10+1;'
                        'cap ?')
models = {
    'cnn': cnn_model.CNNModel
}

def get_distributed_batched_dataset(total_batch_size, num_gpus, max_epochs,
                             data_dir, dataset, split='default'):
    """Reads the input data and set the batch_size to 1/`num_gpus`
    of the `total_batch_size`. 

    Args:
        total_batch_size: Total number of data entries over all towers.
        num_gpus: The number of GPUs available to use.
        max_epochs: The maximum epochs to run.
        data_dir: The directory containing the data.
        dataset: The name of dataset.
        split: 'train', 'test', 'default' (only for noise data)
    Returns:
        bat: A list of batched dataset
    """
    with tf.device('/gpu:0'):
        if dataset == 'mnist':
            raise NotImplementedError('mnist not implemented yet!')
        elif dataset == 'norb':
            raise NotImplementedError('norb not implemented yet!')
        elif dataset == 'cifar10':
            distributed_batched_dataset, specs = cifar10_input.inputs(
                split, data_dir, total_batch_size, num_gpus, max_epochs)
        elif dataset == 'noise':
            distributed_batched_noise_images, specs = noise_input_.inputs(
                num_gpus, max_epochs, total_batch_size)
            return distributed_batched_noise_images, specs
        else:
            raise ValueError(
                'Unexpected dataset {} read!'.format(dataset))
    return distributed_batched_dataset, specs

def find_latest_checkpoint_info(load_dir):
    """Finds the latest checkpoint information.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        latest global step, latest checkpoint path, 
    """
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
        latest_step = extract_step(ckpt.model_checkpoint_path)
        return latest_step, ckpt.model_checkpoint_path
    return -1, None

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

def _compute_averaged_activation_grads(num_gpus):
    """Compute the averaged activation grads. This function adds some 
    extra ops to the original graph, namely calculating the gradients of 
    the objective functions w.r.t. input batched_images.

    Args:
        in_ph_tensors: a list of placeholder tensors that reserved
            for batched_images.
        visual_tensors: a list of lists of activation/logits tensors we used 
            to compute objective functions.
    Returns:
        A dictionary whose keys are the name of the target layer and 
        values are the gradients of the objective functions w.r.t.
        the input.
    """
    in_ph_tensors = []
    visual_tensor_lists = [[] for _ in range(num_gpus)]
    for j in range(num_gpus):
        in_ph_tensors.append(tf.get_collection('tower_%d_placeholders' % j)[0])
        for k, act_tensor in enumerate(tf.get_collection('tower_%d_visual' % j)):
            visual_tensor_lists[k].append(act_tensor)
    
    concated_ph_tensor = tf.concat(in_ph_tensors, 0)
    concated_visual_tensors = [tf.concat(ts) for ts in visual_tensor_lists]
    print(concated_ph_tensor.shape)
    print([ts.shape for ts in concated_visual_tensors])

def run_visual_session(iterator, specs, num_gpus, summary_dir):
    """

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus.
        summary_dir: str, directory storing the checkpoints.
    """
    # Find latest checkpoint information
    latest_step, latest_ckpt_path = find_latest_checkpoint_info(summary_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # Restore variables
        saver.restore(sess, latest_ckpt_path)

        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        _compute_averaged_activation_grads(num_gpus)
        
        # while True: # epoch loop
        #     try:
        #         batch_vals = []
        #         for j in range(num_gpus):
        #             batch_vals.append(sess.run(batch_data))

        #         # Get placeholders and create feed_dict
        #         feed_dict = {}
        #         placeholders = tf.get_collection('placeholders')
        #         for j, batch_val in enumerate(batch_vals):
        #             for ph in placeholders:
        #                 if 'tower_%d' % j in ph.name:
        #                     if 'batched_images' in ph.name:
        #                         feed_dict[ph] = batch_val['images']
                
        #         results = tf.get_collection('results')[0] # namedtuple
                

        #         pass
        #     except tf.errors.OutOfRangeError:
        #         break


def visual(hparams, dataset, model_type,
           total_batch_size, num_gpus, summary_dir, 
           n_repeats):
    """Visualize available layers given noise images.

    Args:
        hparams:
        dataset:
        model_type:
        total_batch_size:
        num_gpus:
        summary_dir:
        n_repeats:
    """
    load_dir = summary_dir + '/visual/'
    summary_dir += '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        distributed_batched_noise_images, dataset_specs = get_distributed_batched_dataset(
            total_batch_size, num_gpus, n_repeats, None, 'noise')
        iterator = distributed_batched_noise_images.make_initializable_iterator()
        # Call visual experiment
        run_visual_session(iterator, dataset_specs, num_gpus, summary_dir)

def run_test_session(iterator, specs, num_gpus, load_dir, summary_dir):
    """

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus.
        summary_dir: str, directory storing the checkpoints.
    Raises:
        ckpt files not found.
    """
    # Find latest checkpoint information
    latest_step, latest_ckpt_path = find_latest_checkpoint_info(load_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        print('Found ckpt at step {}'.format(latest_step))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # Restore variables
        saver.restore(sess, latest_ckpt_path)

        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        while True: # epoch loop
            try:
                batch_vals = []
                for j in range(num_gpus):
                    batch_vals.append(sess.run(batch_data))
                
                # Get placeholders and create feed_dict
                feed_dict = {}
                for j, batch_val in enumerate(batch_vals):
                    placeholders = tf.get_collection('tower_%d_placeholders' % j)
                    for ph in placeholders:
                        if 'batched_images' in ph.name:
                            feed_dict[ph] = batch_val['images']
                        elif 'batched_labels' in ph.name:
                            feed_dict[ph] = batch_val['labels']

                res_acc = tf.get_collection('accuracy')[0]
                
                accuracy = sess.run(
                    res_acc,
                    feed_dict=feed_dict)
                print('accuracy {0:.4f}'.format(accuracy))
            except tf.errors.OutOfRangeError:
                break

def test(hparams, data_dir, dataset, model_type, total_batch_size,
                  num_gpus, summary_dir, max_to_keep):
    """Restore the graph and variables, and evaluate the the metrics.

    Args:
        hparams: The hyperparameters to build the model graph.
        data_dir: The directory containing the input data.
        dataset: The name of the dataset for the experiments.
        model_type: The name of the model architecture.
        total_batch_size: Total batch size, will be distributed to `num_gpus` GPUs.
        num_gpus: The number of GPUs available.
        summary_dir: The directory to write summaries and save the model.
        max_to_keep: Maximum checkpoint files to keep.
    """
    load_dir = summary_dir + '/train/'
    summary_dir += '/test/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        distributed_batched_dataset, dataset_specs = get_distributed_batched_dataset(
            total_batch_size, num_gpus, 1, data_dir, dataset, 'test')
        iterator = distributed_batched_dataset.make_initializable_iterator()
        # Call test experiment
        run_test_session(iterator, dataset_specs, num_gpus, load_dir, summary_dir)
    pass

def run_train_session(iterator, specs, num_gpus, # Dataset related
                      summary_dir, max_to_keep, # Checkpoint related
                      result, save_epochs, max_epochs): # Model related
    """Starts a session, 

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus.
        summary_dir: str, directory storing the checkpoints.
        max_to_keep: scalar, maximum number of ckpt to keep.
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
        saver = tf.train.Saver(max_to_keep=max_to_keep)

        epoch_time = 0
        total_time = 0
        step_counter = 0
        # Start feeding process
        while True: # epoch loop
            start_anchor = time.time() # time anchor
            step_counter += 1
            
            try:
                batch_vals = []
                for j in range(num_gpus): # GPU loop
                    batch_vals.append(sess.run(batch_data))
                
                # Get placeholders and create feed_dict
                feed_dict = {}
                for j, batch_val in enumerate(batch_vals):
                    placeholders = tf.get_collection('tower_%d_placeholders' % j)
                    for ph in placeholders:
                        if 'batched_images' in ph.name:
                            feed_dict[ph] = batch_val['images']
                        elif 'batched_labels' in ph.name:
                            feed_dict[ph] = batch_val['labels']
                summary, accuracy, _ = sess.run(
                    [result.summary, result.accuracy, result.train_op],
                    feed_dict=feed_dict)
                """Add summary"""
                writer.add_summary(summary, global_step=step_counter)
                time_consuming = time.time() - start_anchor
                epoch_time += time_consuming 
                total_time += time_consuming
                """Save ckpts"""
                if step_counter % (specs['steps_per_epoch'] * save_epochs) == 0:
                    ckpt_path = saver.save(
                        sess, os.path.join(summary_dir, 'model.ckpt'),
                        global_step=step_counter)
                    print("{0} epochs done (step = {1}), accuracy {2:.4f}. {3:.2f}s, checkpoint saved at {4}".format(
                        step_counter // specs['steps_per_epoch'], 
                        step_counter, 
                        accuracy, 
                        epoch_time, 
                        ckpt_path))
                    epoch_time = 0
                elif step_counter % specs['steps_per_epoch'] == 0:
                    print("{0} epochs done (step = {1}), accuracy {2:.4f}. {3:.2f}s".format(
                        step_counter // specs['steps_per_epoch'], 
                        step_counter, 
                        accuracy, 
                        epoch_time))
                    epoch_time = 0
            except tf.errors.OutOfRangeError:
                break
            # Finished one step
        print('total time: {0}:{1}:{2}, accuracy: {3:.4f}.'.format(
            int(total_time // 3600), 
            int(total_time % 3600 // 60), 
            int(total_time % 60),
            accuracy))
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
                   num_gpus, summary_dir, max_to_keep,
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
        max_to_keep: Maximum checkpoint files to keep.
        save_epochs: How often the training model should be saved.
        max_epochs: Maximum epochs to train.
    """
    summary_dir += '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default():
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
            total_batch_size, num_gpus, max_epochs, data_dir, dataset, 'train')
        iterator = distributed_batched_dataset.make_initializable_iterator()
        # Initialize model with hparams and dataset_specs
        model = models[model_type](hparams, dataset_specs)
        # Build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        result, _ = model.build_model_on_multi_gpus(num_gpus)
        """Print stats
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        """
        # Clear summary directory, TODO: start train from where left.
        # Call train experiment
        run_train_session(iterator, dataset_specs, num_gpus,
                          summary_dir, max_to_keep,
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
                       FLAGS.num_gpus, FLAGS.summary_dir, FLAGS.max_to_keep,
                       FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'test':
        test(hparams, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size,
                      FLAGS.num_gpus, FLAGS.summary_dir, FLAGS.max_to_keep)
    elif FLAGS.mode == 'visual':
        visual(hparams, FLAGS.dataset, FLAGS.model,
               FLAGS.total_batch_size, FLAGS.num_gpus, FLAGS.summary_dir,
               FLAGS.n_repeats)
    elif FLAGS.mode == 'dream':
        pass
    else:
        raise NotImplementedError(
            "No matching mode found for '{}'".format(FLAGS.mode))
    


if __name__ == '__main__':
    tf.app.run()
