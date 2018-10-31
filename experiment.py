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

tf.flags.DEFINE_integer('num_gpus', 1,
                        'Number of GPUs to use.')
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
tf.flags.DEFINE_integer('total_batch_size', 100, 
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
                        'naive = 1;\n'
                        'multiscale = 1;\n'
                        'dream = # of different samples for each class')
tf.flags.DEFINE_integer('iter_n', 10,
                        'Number of iteration to run the gradient ascent')
tf.flags.DEFINE_string('step', '1.0',
                       'Size of step for each iteration')
tf.flags.DEFINE_string('threshold', '0.0',
                       'Those gradients after divided by the its standard deviations that larger than the threshold will be added')
models = {
    'cnn': cnn_model.CNNModel,
    'cap': capsule_model.CapsuleModel
}

def get_distributed_dataset(total_batch_size, num_gpus, 
                            max_epochs, data_dir, dataset, 
                            split='default', n_repeat=None):
    """Reads the input data from input_data functions.

    For 'train' and 'test' splits,
        given `num_gpus` GPUs and `total_batch_size`, assert 
        `total_batch_size` % `num_gpus` == 0, we distribute 
        those `total_batch_size` into `num_gpus` partitions,
        denoted as `batch_size`, otherwise raise error.
    
    TODO: do it when reach visualization part
    For 'noise' and 'dream' splits,
        check if `total_batch_size` ≡ 1, otherwise raise 'ValueError'.
        In this case, we will duplicate every example `num_gpus` times
        so that when we pass the examples into the multi-tower models,
        it is calculating and averaging the gradients of the same images.

    Args:
        total_batch_size: total number of data entries over all towers.
        num_gpus: number of GPUs available to use.
        max_epochs: for 'train' split, this parameter decides the number of 
            epochs to train for the model; for 'test' split, this parameter
            should ≡ 1 since we are not doing resemble evalutions in this project.
        data_dir: the directory containing the data.
        dataset: the name of dataset.
        split: 'train', 'test', 'noise', 'dream'.
        n_repeat('noise' and 'dream'): the number of repeats of the same image.
    Returns:
        batched_dataset: Dataset object.
        specs: dataset specifications.
    """
    with tf.device('/gpu:0'):
        if split == 'train' or split == 'test':
            assert total_batch_size % num_gpus == 0
            if dataset == 'mnist':
                distributed_dataset, specs = mnist_input.inputs(
                    total_batch_size, num_gpus, max_epochs,
                    data_dir, split)
            elif dataset == 'fashion_mnist': # TODO
                raise NotImplementedError('fashsion_mnist not implemented yet.')
            elif dataset == 'cifar10': # TODO
                raise NotImplementedError('cifar10 not implemented yet.')
        elif split == 'noise':
            if dataset == 'mnist':
                dataset, specs = noise_input_.inputs(
                    max_epochs, n_repeat, depth=1)
            elif dataset == 'fashion_mnist': # TODO
                raise NotImplementedError('')
            elif dataset == 'cifar10': # TODO
                raise NotImplementedError('')
        elif split == 'dream':
            if dataset == 'mnist': # TODO
                raise NotImplementedError('')
            elif dataset == 'fashion_mnist': # TODO
                raise NotImplementedError('')
            elif dataset == 'cifar10': # TODO
                raise NotImplementedError('')
        else:
            raise ValueError('')
    return distributed_dataset, specs

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

def _compute_activation_grads(tower_idx):
    """Compute the averaged activation grads. Because the weights are 
    shared among towers, so we simply take one tower to compute the 
    gradients instead of calculating the gradients of all the towers. 
    This function adds some extra ops to the original graph, namely 
    calculating the gradients of the objective functions w.r.t. 
    input batched_images.
    
    Args:
        tower_idx: the index number for this tower. Each tower is named
            as tower_{tower_idx} and resides on gpu:{tower_idx}.
    Returns:
        A dictionary whose keys are the name of the target layer and 
        values are the gradients of the objective functions w.r.t.
        the input.
    """
    # get batched_images placeholder tensor
    batched_images_t = tf.get_collection('tower_%d_batched_images' % tower_idx)[0]

    visual_tensors = tf.get_collection('tower_%d_visual' % tower_idx)
                                                 # returns a list of k logits tensors,
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
            # ch_t_obj = tf.reduce_mean(ch_t, name=ch_t_name+'/obj')
            ch_t_obj = ch_t
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

def _write_specs_file(write_dir, vis_or_dream_type, dataset, total_batch_size,
                      max_epochs, iter_n, step, threshold):
    write_dir = os.path.join(write_dir, 'max_ep_{}-iter_n_{}-step_{}-th_{}'.format(
        max_epochs, iter_n, step, threshold))
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    with open(os.path.join(write_dir, 'specs.txt'), 'w+') as f:
        f.write('type: {};\n'.format(vis_or_dream_type))
        f.write('dataset: {};\n'.format(dataset))
        f.write('total_batch_size: {};\n'.format(total_batch_size))
        f.write('max_epochs: {};\n'.format(max_epochs))
        f.write('iter_n: {};\n'.format(iter_n))
        f.write('step: {};\n'.format(step))
        f.write('threshold: {};\n'.format(threshold))
    return write_dir

def run_visual_session(num_gpus, total_batch_size, max_epochs, data_dir, dataset,
                       iter_n, step, threshold,
                       load_dir, summary_dir, vis_or_dream_type='naive'):
    """Start visualization session. Producing visualization results to summary_dir.

    Args:
        num_gpus: number of GPUs available to use.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        max_epochs: maximum epochs to train.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        load_dir: the directory to load the model.
        summary_dir: the directory to write the files.
        vis_or_dream_type: 'naive', 'multiscale', 'pyramid' (TODO) or 'dream'
    """
    # Init writing directory
    write_dir = os.path.join(summary_dir, vis_or_dream_type)
    write_dir = _write_specs_file(write_dir, vis_or_dream_type, dataset, total_batch_size,
                                  max_epochs, iter_n, step, threshold)

    # Find latest checkpoint information
    latest_step, latest_ckpt_path = find_latest_checkpoint_info(load_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph and restore variables
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        saver.restore(sess, latest_ckpt_path)

        # Calculate the gradients
        result_grads = _compute_activation_grads(tower_idx=0)
        num_batches_per_epoch = len(result_grads) # number of batches per epoch
        print('Number of gradients computed (= number of batches per epoch): ', 
              num_batches_per_epoch)

        # Get batched dataset and declare initializable iterator
        possible_vis_types = ['naive', 'multiscale', 'pyramid']
        possible_dream_types = ['dream']
        if vis_or_dream_type in possible_vis_types:
            split = 'noise'
        elif vis_or_dream_type in possible_dream_types:
            split = 'dream'
        else:
            raise ValueError(
                "{} is not one of 'naive', 'multiscale', 'pyramid' or 'dream'".format(vis_or_dream_type))
        batched_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, 
            max_epochs, data_dir, dataset, 
            split=split, n_repeat=num_batches_per_epoch)
        iterator = batched_dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        # batch_size ≡ 1
        # Visuals (naive, multiscale, pyramid): 
        #                 max_epochs * steps_per_epoch(=num_batches_per_epoch)
        # Dream: 
        #   num_classes * max_epochs * steps_per_epoch(=num_batches_per_epoch)
        #
        if split == 'noise':
            num_class_loop = 1
        elif split == 'dream':
            num_class_loop = specs['num_classes']
        else:
            raise ValueError("'split' value invalid")

        for i in range(max_epochs):
            for j in range(num_class_loop):
                for k in range(num_batches_per_epoch):
                    try:
                        # Get feed dict reference
                        placeholders = tf.get_collection('placeholders')
                        for ph in placeholders:
                            if 'batched_images' in ph.name:
                                ph_ref = ph
                        # Get one batch values
                        batch_val = sess.run(batch_data)
                        # Run different experiments
                        if vis_or_dream_type == 'naive':
                            layer_visual.render_naive(
                                t_grad=result_grads[k], img0=batch_val['images'],
                                in_ph_ref=ph_ref, sess=sess, write_dir=write_dir,
                                iter_n=iter_n, step=step)
                        elif vis_or_dream_type == 'multiscale':
                            layer_visual.render_multiscale(
                                t_grad=result_grads[k], img0=batch_val['images'],
                                in_ph_ref=ph_ref, sess=sess, write_dir=write_dir)
                        elif vis_or_dream_type == 'pyramid':
                            raise NotImplementedError('pyramid not implemented!')
                        elif vis_or_dream_type == 'dream':
                            lbl = batch_val['labels']
                            layer_visual.render_naive(
                                t_grad=result_grads[k], img0=batch_val['images'],
                                in_ph_ref=ph_ref, sess=sess, write_dir=write_dir, 
                                iter_n=iter_n, step=step, threshold=threshold, ep_i=i, lbl=lbl)
                        else:
                            raise ValueError("mode type is not one of 'train', 'test', 'naive', 'multiscale', 'pyramid', or 'dream'!")
                        print('\n{0} {1} {0} total:class:gradient = {2:.1f}% ~ {3:.1f}% ~ {4:.1f}%'.format(
                            ' '*3, '-'*5, 
                            100.0*(i * num_class_loop * num_batches_per_epoch + j * num_batches_per_epoch + k + 1) / (max_epochs * num_class_loop * num_batches_per_epoch),
                            100.0*(j * num_batches_per_epoch + k + 1)/(num_class_loop * num_batches_per_epoch),
                            100.0*(k + 1)/num_batches_per_epoch))
                    except tf.errors.OutOfRangeError:
                        break

def visual(num_gpus, data_dir, dataset,
           total_batch_size, summary_dir, max_epochs, 
           iter_n, step, threshold, vis_or_dream_type='naive'):
    """Visualize available layers given noise images.

    Args:
        num_gpus: number of GPUs available to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: the directory to write summaries and save the model.
        max_epochs: maximum epochs to train.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        vis_or_dream_type: 'naive', 'multiscale', 'pyramid' (TODO) or 'dream'
    """
    load_dir = summary_dir + '/train/'
    summary_dir += '/visual/'
    # Declare the empty model graph
    with tf.Graph().as_default():
        # Call visual experiment
        run_visual_session(num_gpus, total_batch_size, max_epochs, data_dir, dataset,
                           iter_n, step, threshold,
                           load_dir, summary_dir, vis_or_dream_type)

def run_test_session(iterator, specs, load_dir):
    """Find latest checkpoint and load the graph and variables.

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        load_dir: str, directory that contains checkpoints.
    Raises:
        ckpt files not found.
    """

    """Load latest checkpoint"""
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
                # Get placeholders and create feed dict
                feed_dict = {}
                try:
                    for i in range(specs['num_gpus']):
                        batch_val = sess.run(batch_data)
                        feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                        feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                except:
                    raise IndexError('index out of range')
                    
                # Get accuracy tensor
                res_acc = tf.get_collection('accuracy')[0]
                # Calculate one total batch accuracy
                accuracy = sess.run(
                    res_acc,
                    feed_dict=feed_dict)
                # Append to the accuracy list.
                accs.append(accuracy)
            except tf.errors.OutOfRangeError:
                break    
        print('accuracy {0:.4f}'.format(np.mean(accs)))

def test(num_gpus, data_dir, dataset, model_type, total_batch_size,
         summary_dir, max_epochs):
    """Restore the graph and variables, and evaluate the the metrics.

    Args:
        num_gpus: number of GPUs available to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        model_type: the name of the model architecture.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: the directory to load the model.
        max_epochs: maximum epochs to evaluate, ≡ 1.
    """
    load_dir = summary_dir + '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        distributed_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, data_dir, dataset, 'test')
        iterator = distributed_dataset.make_initializable_iterator()
        # Call test experiment
        run_test_session(iterator, specs, load_dir)

def run_train_session(iterator, specs, # Dataset related
                      summary_dir, max_to_keep, # Checkpoint related
                      joined_result, save_epochs): # Model related
    """Starts a session, train the model, write summary into event file,
    and save the whole graph every {save_epochs} epochs.

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        num_gpus: scalar, number of gpus.
        summary_dir: str, directory storing the checkpoints.
        max_to_keep: scalar, maximum number of ckpt to keep.
        joined_result: namedtuple, TowerResult('inferred', 'train_op', 
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
                # Get placeholders and create feed_dict
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                    feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']

                summary, accuracy, _ = sess.run(
                    [joined_result.summary, joined_result.accuracy, joined_result.train_op],
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
                        step_counter // specs['steps_per_epoch'] + 1,
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

def train(hparams, num_gpus, data_dir, dataset, model_type, total_batch_size,
                   summary_dir, max_to_keep,
                   save_epochs, max_epochs):
    """Trains a model with batch sizes of 100 to 50000/100*`max_epochs` steps.

    It will initialize the model with either previously saved model in the 
    `summary_dir` or start from scratch if the directory is empty.
    The training is distributed on `num_gpus` GPUs. It writes a summary at 
    every step and saves the model every `save_epochs` steps.

    Args:
        hparams: the hyperparameters to build the model graph.
        num_gpus: number of GPUs to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        model_type: the name of the model architecture.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: the directory to write summaries and save the model.
        max_to_keep: maximum checkpoint files to keep.
        save_epochs: how often the training model should be saved.
        max_epochs: maximum epochs to train.
    """
    summary_dir += '/train/'

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        distributed_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, data_dir, dataset, 'train')
        iterator = distributed_dataset.make_initializable_iterator()
        # Initialize model with hparams and specs
        model = models[model_type](hparams, specs)
        # Build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        joined_result = model.build_model_on_single_gpu()

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
                          joined_result, save_epochs)

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
        train(hparams, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size,
                       FLAGS.summary_dir, FLAGS.max_to_keep,
                       FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'test':
        test(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size,
             FLAGS.summary_dir, FLAGS.max_epochs)
    elif FLAGS.mode == 'naive' or FLAGS.mode == 'multiscale' or FLAGS.mode == 'pyramid' or FLAGS.mode == 'dream': # TODO
        visual(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model,
               FLAGS.total_batch_size, FLAGS.summary_dir, FLAGS.max_epochs, 
               FLAGS.iter_n, float(FLAGS.step), float(FLAGS.threshold), FLAGS.mode)
    else:
        raise ValueError(
            "No matching mode found for '{}'".format(FLAGS.mode))
    
if __name__ == '__main__':
    tf.app.run()
