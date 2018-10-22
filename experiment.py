# Copyright 2017 The TensorFlow Authors All Rights Reserved.
# Copyright 2018 Xu Chen All Rights Reserved.
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
import re
from io import BytesIO
import PIL.Image

import numpy as np 
import tensorflow as tf 

from input_data.mnist import mnist_input
from input_data.cifar10 import cifar10_input
from input_data.noise import noise_input_
from models import cnn_model
from models import capsule_model
from dream import layer_visual

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', 'train',
                       'train, test, naive, multiscale, dream')
tf.flags.DEFINE_string('hparams_override', None,
                        '--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false')
tf.flags.DEFINE_string('data_dir', None, 
                       'The data directory.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'The dataset to use for the experiment.\n'
                       'mnist, cifar10.')
tf.flags.DEFINE_string('model', 'cnn',
                       'The model to use for the experiment.\n'
                       'cap or cnn')
tf.flags.DEFINE_integer('batch_size', 100, 
                        'Total batch size.')
tf.flags.DEFINE_string('summary_dir', './summary',
                       'Main directory for the experiments.')
tf.flags.DEFINE_integer('max_to_keep', None, 
                        'Maximum number of checkpoint files to keep.')
tf.flags.DEFINE_integer('save_epochs', 5, 'How often to save checkpoints.')
tf.flags.DEFINE_integer('max_epochs', 20, 
                        'Number of epochs to train, test, naive, multiscale or dream.\n'
                        'train ~ 1000 (20 when debugging);\n'
                        'test = 1;\n'
                        'naive > 2000 (cnn), > ? (cap);\n'
                        'multiscale ~ 5x naive;\n'
                        'dream ∝ # of given image')
models = {
    'cnn': cnn_model.CNNModel,
    'cap': capsule_model.CapsuleModel
}

def get_batched_dataset(batch_size, max_epochs,
                        data_dir, dataset, split='default'):
    """Reads the input data and set the batch_size to 1/`num_gpus`
    of the `batch_size`. 

    Args:
        batch_size: Total number of data entries over all towers.
        max_epochs: The maximum epochs to run.
        data_dir: The directory containing the data.
        dataset: The name of dataset.
        split: 'train', 'test', 'default' (only for noise data)
    Returns:
        bat: A list of batched dataset
    """
    with tf.device('/gpu:0'):
        if split == 'noise':
            if dataset == 'cifar10':
                batched_dataset, specs = noise_input_.inputs(
                    n_repeats=max_epochs)
            elif dataset == 'mnist':
                batched_dataset, specs = noise_input_.inputs(
                    n_repeats=max_epochs,
                    depth=1)
        else:
            if dataset == 'mnist':
                batched_dataset, specs = mnist_input.inputs(
                    split=split,
                    data_dir=data_dir,
                    batch_size=batch_size,
                    max_epochs=max_epochs)
            elif dataset == 'cifar10':
                batched_dataset, specs = cifar10_input.inputs(
                    split=split,
                    data_dir=data_dir,
                    batch_size=batch_size,
                    max_epochs=max_epochs)
            else:
                raise NotImplementedError('Unexpected dataset read!!')
    return batched_dataset, specs

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

def _compute_activation_grads():
    """Compute the averaged activation grads. This function adds some 
    extra ops to the original graph, namely calculating the gradients of 
    the objective functions w.r.t. input batched_images.

    Returns:
        A dictionary whose keys are the name of the target layer and 
        values are the gradients of the objective functions w.r.t.
        the input.
    """
    ph_tensors = tf.get_collection('placeholders') # returns a list of two placeholder tensors,
                                                   # contain 'images' and 'labels'
    # get input 'batched_images' tensor
    for ph in ph_tensors:
        if 'batched_images' in ph.name:
            batched_images_t = ph

    visual_tensors = tf.get_collection('visual') # returns a list of k logits tensors,
                                                 # tensors before activation function.
    for vt in visual_tensors:
        print('vt name: ', vt.name)
    result_grads = []
    for logit_t in visual_tensors[-1:]:
        # write tensor prefix name
        logit_t_name_prefix = '/'.join(logit_t.name.split('/')[:-1]) + '/' \
                              + logit_t.name.split('/')[-1][:-2]
        print(logit_t_name_prefix)
        print(logit_t.get_shape()) # (?, ch, ...)
        # split the tensor by channels, a list containing `ch` number of tensors 
        # each having the shape of (?, 1, ...)
        splited_logit_t_by_chs = tf.split(logit_t, num_or_size_splits=logit_t.get_shape()[1],
                                          axis=1, name=logit_t_name_prefix + '/split_op')
        """
        last_ch_t_name= '_'.join(splited_logit_t_by_chs[-1].name.split(':'))
        last_ch_obj = tf.reduce_mean(splited_logit_t_by_chs[-1], name=last_ch_t_name+'/obj')
        last_ch_grads = tf.gradients(last_ch_obj, batched_images_t, name=last_ch_t_name+'/grads')
        result_grads.append(last_ch_grads)
        print(splited_logit_t_by_chs[-1], last_ch_obj, batched_images_t, last_ch_grads)
        """
        # take first 5 splited channels
        splited_logit_t_by_chs = splited_logit_t_by_chs

        for ch_idx, ch_t in enumerate(splited_logit_t_by_chs):
            ch_t_name = '_'.join(ch_t.name.split(':'))
            ch_t_obj = tf.reduce_mean(ch_t, name=ch_t_name+'/obj')
            ch_t_grads = tf.gradients(ch_t_obj, batched_images_t, name='gradients/' + ch_t_name)
            #ch_t_grads = tf.gradients(ch_t_obj, batched_images_t)
            result_grads.append(ch_t_grads)
            # print(ch_t, ch_t_obj, batched_images_t, ch_t_grads)
            print('Done processing {0} ---- {1:.2f}%'.format(
                ch_t_name, (1+ch_idx)*100/float(len(splited_logit_t_by_chs))))
        print("")
        
    print('Gradients computing completed!')
    # flatten the list
    result_grads = [item for sub in result_grads for item in sub]
    
    return result_grads

def run_visual_session(iterator, specs, load_dir, summary_dir, vis_type='naive'):
    """Start visualization session. Producing results to summary_dir.

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        load_dir: str, directory that contains checkpoints.
        summary_dir: str, directory storing the resultant images.
        vis_type: 'naive', 'multiscale', 'pyramid' (TODO)
    """
    # Find latest checkpoint information
    latest_step, latest_ckpt_path = find_latest_checkpoint_info(load_dir)
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
        # Calculate the gradients
        result_grads = _compute_activation_grads()
        print('Number of gradients computed: ', len(result_grads))

        for t_idx, t_grad in enumerate(result_grads):
            try:
                batched_images = sess.run(batch_data)

                placeholders = tf.get_collection('placeholders')
                for ph in placeholders:
                    if 'batched_images' in ph.name:
                        ph_ref = ph 
                if vis_type == 'naive':
                    layer_visual.render_naive(t_grad, batched_images, ph_ref, sess, summary_dir)
                elif vis_type == 'multiscale':
                    layer_visual.render_multiscale(t_grad, batched_images, ph_ref, sess, summary_dir)
                elif vis_type == 'pyramid':
                    raise NotImplementedError('pyramid not implemented!')
                else:
                    raise ValueError("mode type is not one of 'train', 'test', 'naive', 'multiscale', 'pyramid', or 'dream'!")
                print('\n{0} {1} {0} {2}%'.format(' '*3, '-'*5, (1+t_idx)*100.0/len(result_grads)))
            except tf.errors.OutOfRangeError:
                break

def visual(hparams, dataset, model_type,
           batch_size, summary_dir, 
           max_epochs, vis_type='naive'):
    """Visualize available layers given noise images.

    Args:
        hparams: The hyperparameters to build the model graph.
        dataset: The name of the dataset for the experiments.
        model_type: The name of the model architecture.
        batch_size: Total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: The directory to write summaries and save the model.
        max_epochs: Maximum epochs to train.
        vis_type: 'naive', 'multiscale', 'pyramid' (TODO)
    """
    load_dir = summary_dir + '/train/'
    summary_dir += '/visual/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        batched_dataset, specs = get_batched_dataset(
            batch_size=batch_size,
            max_epochs=max_epochs,
            data_dir=None,
            dataset=dataset,
            split='noise')
        iterator = batched_dataset.make_initializable_iterator()
        # Call visual experiment
        run_visual_session(iterator, specs, load_dir, summary_dir, vis_type)

def run_test_session(iterator, specs, load_dir, summary_dir):
    """Find latest checkpoint and load the graph and variables.

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        load_dir: str, directory that contains checkpoints.
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
        accs = []

        while True: # epoch loop
            try:
                batch_val = sess.run(batch_data)
                
                # Get placeholders and create feed dict
                feed_dict = {}
                placeholders = tf.get_collection('placeholders')
                for ph in placeholders:
                    if 'batched_images' in ph.name:
                        feed_dict[ph] = batch_val['images']
                    elif 'batched_labels' in ph.name:
                        feed_dict[ph] = batch_val['labels']

                res_acc = tf.get_collection('accuracy')[0]
                
                accuracy = sess.run(
                    res_acc,
                    feed_dict=feed_dict)
                accs.append(accuracy)
            except tf.errors.OutOfRangeError:
                break    
        print('accuracy {0:.4f}'.format(np.mean(accs)))

def test(hparams, data_dir, dataset, model_type, batch_size,
                  summary_dir, max_to_keep, max_epochs):
    """Restore the graph and variables, and evaluate the the metrics.

    Args:
        hparams: The hyperparameters to build the model graph.
        data_dir: The directory containing the input data.
        dataset: The name of the dataset for the experiments.
        model_type: The name of the model architecture.
        batch_size: Total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: The directory to write summaries and save the model.
        max_to_keep: Maximum checkpoint files to keep.
        max_epochs: Maximum epochs to train.
    """
    load_dir = summary_dir + '/train/'
    summary_dir += '/test/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        batched_dataset, specs = get_batched_dataset(
            batch_size=batch_size,
            max_epochs=max_epochs,
            data_dir=data_dir,
            dataset=dataset,
            split='test')
        iterator = batched_dataset.make_initializable_iterator()
        # Call test experiment
        run_test_session(iterator, specs, load_dir, summary_dir)

def run_train_session(iterator, specs, # Dataset related
                      summary_dir, max_to_keep, # Checkpoint related
                      tower_output, save_epochs): # Model related
    """Starts a session, 

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus.
        summary_dir: str, directory storing the checkpoints.
        max_to_keep: scalar, maximum number of ckpt to keep.
        tower_output: namedtuple, TowerResult('inferred', 'train_op', 
                                    'summary', 'correct', 'accuracy')
        save_epochs: scalar, how often to save the model
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
                # batch_vals = []
                # for j in range(num_gpus): # GPU loop
                #     batch_vals.append(sess.run(batch_data))
                batch_val = sess.run(batch_data)

                # Get placeholders and create feed_dict
                feed_dict = {}
                placeholders = tf.get_collection('placeholders')
                for ph in placeholders:
                    if 'batched_images' in ph.name:
                        feed_dict[ph] = batch_val['images']
                    elif 'batched_labels' in ph.name:
                        feed_dict[ph] = batch_val['labels']
                
                summary, accuracy, _ = sess.run(
                    [tower_output.summary, tower_output.accuracy, tower_output.train_op],
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
                else:
                    print("running {0} epochs {1:.1f}%, total time ~ {2}:{3}:{4}".format(
                        step_counter // specs['steps_per_epoch'],
                        step_counter % specs['steps_per_epoch'] * 100.0 / specs['steps_per_epoch'],
                        int(total_time // 3600), 
                        int(total_time % 3600 // 60), 
                        int(total_time % 60)),
                        end='\r')
            except tf.errors.OutOfRangeError:
                break
            # Finished one step
        print('total time: {0}:{1}:{2}, accuracy: {3:.4f}.'.format(
            int(total_time // 3600), 
            int(total_time % 3600 // 60), 
            int(total_time % 60),
            accuracy))

def train(hparams, data_dir, dataset, model_type, batch_size,
                   summary_dir, max_to_keep,
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
        batch_size: Total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: The directory to write summaries and save the model.
        max_to_keep: Maximum checkpoint files to keep.
        save_epochs: How often the training model should be saved.
        max_epochs: Maximum epochs to train.
    """
    summary_dir += '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        batched_dataset, specs = get_batched_dataset(
            batch_size=batch_size,
            max_epochs=max_epochs,
            data_dir=data_dir,
            dataset=dataset,
            split='train')
        iterator = batched_dataset.make_initializable_iterator()
        # Initialize model with hparams and specs
        model = models[model_type](hparams, specs)
        # Build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        tower_output = model.build_model_on_single_gpu()
        # TODO here
        """Print stats"""
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        """"""
        # Clear summary directory, TODO: start train from where left.
        # Call train experiment
        run_train_session(iterator, specs,
                          summary_dir, max_to_keep,
                          tower_output, save_epochs)

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
        train(hparams, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.batch_size,
                       FLAGS.summary_dir, FLAGS.max_to_keep,
                       FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'test':
        test(hparams, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.batch_size,
                      FLAGS.summary_dir, FLAGS.max_to_keep, FLAGS.max_epochs)
    elif FLAGS.mode == 'naive' or FLAGS.mode == 'multiscale' or FLAGS.mode == 'pyramid':
        visual(hparams, FLAGS.dataset, FLAGS.model,
               FLAGS.batch_size, FLAGS.summary_dir,
               FLAGS.max_epochs, FLAGS.mode)
    elif FLAGS.mode == 'dream':
        raise NotImplementedError('dream not implemented!')
    else:
        raise ValueError(
            "No matching mode found for '{}'".format(FLAGS.mode))
    
if __name__ == '__main__':
    tf.app.run()
