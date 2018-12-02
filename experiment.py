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
import glob
import numpy as np 
import tensorflow as tf 

from input_data.mnist import mnist_input, mnist_dream_inputs
from input_data.fashion_mnist import fashion_mnist_input, fashion_mnist_dream_input
from input_data.svhn import svhn_input, svhn_dream_input
from input_data.cifar10 import cifar10_input, cifar10_dream_input
from input_data.noise import noise_dream_input
from models import cnn_model
from models import capsule_model
from grad import naive_max_norm, max_norm_diff, naive_max_caps_dim, max_caps_dim_diff, utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_gpus', 2,
                        'Number of GPUs to use.')
tf.flags.DEFINE_string('mode', 'train',
                       'train: train the model;\n'
                       'evaluate: evaluate the model for both training and testing set;\n'
                       'Capsule Norm Aspect: \n'
                       '    naive_max_norm, max_norm_diff, noise_naive_max_norm, noise_max_norm_diff;\n'
                       'Capsule Direction Aspect:\n'
                       '    naive_max_caps_dim, max_caps_dim_diff, noise_naive_max_caps_dim, noise_max_caps_dim_diff.')
tf.flags.DEFINE_string('hparams_override', None,
                        '--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false')
tf.flags.DEFINE_string('data_dir', None, 
                       'The data directory.')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.\n'
                       'mnist, fashion_mnist, svhn, cifar10.')
tf.flags.DEFINE_string('model', 'cap',
                       'The model to use for the experiment.\n'
                       'cap or cnn')
tf.flags.DEFINE_integer('total_batch_size', 1, 
                        'Total batch size.')
tf.flags.DEFINE_string('summary_dir', './summary',
                       'Main directory for the experiments.')
tf.flags.DEFINE_integer('max_to_keep', None, 
                        'Maximum number of checkpoint files to keep.')
tf.flags.DEFINE_integer('save_epochs', 10, 'How often to save checkpoints.')
tf.flags.DEFINE_integer('max_epochs', 10, 
                        'train, evaluate: maximum epochs to run;\n'
                        'others: number of different examples to visualization.')
tf.flags.DEFINE_integer('iter_n', 1000,
                        'Number of iteration to run the gradient ascent\n'
                        'the code only record any iterations in\n'
                        '[1, 2, 3, 4, 5, 6, 7, 8, 9,\n'
                        ' 10, 20, 40, 60, 80,\n'
                        ' 100, 200, 400, 600, 800, 1000]')
tf.flags.DEFINE_string('step', '0.1',
                       'Size of step for each iteration')
tf.flags.DEFINE_string('threshold', '0.0',
                       'Those gradients after divided by the its standard deviations that larger than the threshold will be added')
tf.flags.DEFINE_integer('image_size', 24,
                       'Define the image size for dataset')

models = {
    'cnn': cnn_model.CNNModel,
    'cap': capsule_model.CapsuleModel
}

vis_grad_computer = {
    'naive_max_norm': naive_max_norm, 
    'max_norm_diff': max_norm_diff, 
    'naive_max_caps_dim': naive_max_caps_dim,
    'max_caps_dim_diff': max_caps_dim_diff
}

NORM_ASPECT_TYPES = ['naive_max_norm', 'max_norm_diff']

DIRECTION_ASPECT_TYPES = ['naive_max_caps_dim', 'max_caps_dim_diff']

def get_distributed_dataset(total_batch_size, num_gpus, 
                            max_epochs, data_dir, dataset, cropped_size,
                            split='default', n_repeats=None):
    """Reads the input data from input_data functions.

    For 'train' and 'test' splits,
        given `num_gpus` GPUs and `total_batch_size`, assert 
        `total_batch_size` % `num_gpus` == 0, we distribute 
        those `total_batch_size` into `num_gpus` partitions,
        denoted as `batch_size`, otherwise raise error.
    
    For 'noise' and 'dream' splits,
        check if `total_batch_size` ≡ 1, otherwise raise 'ValueError'.
        In this case, we will duplicate every example `num_gpus` times
        so that when we pass the examples into the multi-tower models,
        it is calculating and averaging the gradients of the same images.

    Args:
        total_batch_size: total number of data entries over all towers;
        num_gpus: number of GPUs available to use;
        max_epochs: for 'train' split, this parameter decides the number of 
            epochs to train for the model; for 'test' split, this parameter
            should ≡ 1 since we are not doing resemble evalutions in this project;
        data_dir: the directory containing the data;
        dataset: the name of dataset;
        cropped_size: image size after cropping;
        split: 'train', 'test', 'noise', 'dream';
        n_repeats('noise' and 'dream'): the number of repeats of the same image.
    Returns:
        batched_dataset: Dataset object.
        specs: dataset specifications.
    """
    with tf.device('/gpu:0'):
        if split == 'train' or split == 'test':
            assert total_batch_size % num_gpus == 0
            if dataset == 'mnist':
                distributed_dataset, specs = mnist_input.inputs(
                    total_batch_size, num_gpus, max_epochs, cropped_size,
                    data_dir, split)
            elif dataset == 'fashion_mnist': 
                distributed_dataset, specs = fashion_mnist_input.inputs(
                    total_batch_size, num_gpus, max_epochs, cropped_size,
                    data_dir, split)
            elif dataset == 'svhn': 
                distributed_dataset, specs = svhn_input.inputs(
                    total_batch_size, num_gpus, max_epochs, cropped_size,
                    data_dir, split)
            elif dataset == 'cifar10':
                distributed_dataset, specs = cifar10_input.inputs(
                    total_batch_size, num_gpus, max_epochs, cropped_size,
                    data_dir, split)
            # the data will be distributed over {num_gpus} GPUs.
            return distributed_dataset, specs
        elif split == 'noise':
            if dataset == 'mnist':
                batched_dataset, specs = noise_dream_input.inputs(
                    'noise', 1, max_epochs, n_repeats, cropped_size)
            elif dataset == 'fashion_mnist': 
                batched_dataset, specs = noise_dream_input.inputs(
                    'noise', 1, max_epochs, n_repeats, cropped_size)
            elif dataset == 'svhn':
                batched_dataset, specs = noise_dream_input.inputs(
                    'noise', 3, max_epochs, n_repeats, cropped_size)
            elif dataset == 'cifar10': 
                batched_dataset, specs = noise_dream_input.inputs(
                    'noise', 3, max_epochs, n_repeats, cropped_size)
            # the data will only have batch_size=1 and not be distributed over {num_gpus} GPUs.
            return batched_dataset, specs
        elif split == 'dream':
            if dataset == 'mnist':
                batched_dataset, specs = mnist_dream_inputs.inputs(
                    'train', data_dir, max_epochs, n_repeats, cropped_size)
            elif dataset == 'fashion_mnist': 
                batched_dataset, specs = fashion_mnist_dream_input.inputs(
                    'train', data_dir, max_epochs, n_repeats, cropped_size)
            elif dataset == 'svhn':
                batched_dataset, specs = svhn_dream_input.inputs(
                    'train', data_dir, max_epochs, n_repeats, cropped_size)
            elif dataset == 'cifar10': 
                batched_dataset, specs = cifar10_dream_input.inputs(
                    'train', data_dir, max_epochs, n_repeats, cropped_size)
            # the data will only have batch_size=1 and not be distributed over {num_gpus} GPUs.
            return batched_dataset, specs
        else:
            raise ValueError('')

def find_event_file_path(load_dir):
    """Finds the event file.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        path to the event file.
    """
    fpath_list = glob.glob(os.path.join(load_dir, 'events.*'))
    if len(fpath_list) == 1:
        return fpath_list[0]
    else:
        raise ValueError

def find_latest_checkpoint_info(load_dir, find_all=False):
    """Finds the latest checkpoint information.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        latest global step, latest checkpoint path, step_ckpt pair list
    """
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
        latest_step = extract_step(ckpt.model_checkpoint_path)
        if find_all == True:
            ckpt_paths = glob.glob(os.path.join(load_dir, 'model.ckpt-*.index'))
            pairs = [(int(re.search('\d+', os.path.basename(path)).group(0)), 
                      os.path.join(os.path.dirname(path), os.path.basename(path)[:-6]))
                      for path in ckpt_paths]
            pairs = sorted(pairs, key=lambda pair: pair[0])
        else:
            pairs = []
        return latest_step, ckpt.model_checkpoint_path, pairs
    return -1, None, []

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

def _write_specs_file(write_dir, aspect_type, dataset, total_batch_size, 
                     max_epochs, iter_n, step, threshold):
    write_dir = os.path.join(write_dir, 'max_ep{}-iter_n{}-step{}-th{}'.format(
        max_epochs, iter_n, step, threshold))
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    with open(os.path.join(write_dir, 'specs.txt'), 'w+') as f:
        f.write('explore type: {};\n'.format(aspect_type))
        f.write('dataset: {};\n'.format(dataset))
        f.write('total_batch_size: {};\n'.format(total_batch_size))
        f.write('max_epochs: {};\n'.format(max_epochs))
        f.write('iter_n: {};\n'.format(iter_n))
        f.write('step: {};\n'.format(step))
        f.write('threshold: {};\n'.format(threshold))
    return write_dir

def run_direction_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, cropped_size,
                         iter_n, step, threshold,
                         load_dir, summary_dir, aspect_type):
    """Start norm aspect exploration. Producing results to summary_dir
    
    Args:
        num_gpus: number of GPUs available to use.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        max_epochs: maximum epochs to train.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        cropped_size: image size after cropping.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        load_dir: the directory to load files.
        summary_dir: the directory to write files.
        aspect_type: 'naive_max_caps_dim'
    """
    # Write specs file
    write_dir = _write_specs_file(summary_dir, aspect_type, dataset, total_batch_size,
                                  max_epochs, iter_n, step, threshold)
    
    # Find out to feed in noise of data
    if 'noise_' in aspect_type:
        aspect_type = aspect_type[6:]
        split = 'noise'
    else:
        split = 'dream'

    # Find latest checkpoint information
    latest_step, latest_ckpt_path, _ = find_latest_checkpoint_info(load_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph and restore variables
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        saver.restore(sess, latest_ckpt_path)

        # Compute the gradients
        result_grads, batched_images, caps_norms_tensor= vis_grad_computer[aspect_type].compute_grads(0)
        n_repeats = 16 # 16 dimensional vector
        print('Number of gradients computed: ', len(result_grads))

        # Get batched dataset and specs
        batched_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, cropped_size,
            split=split, n_repeats=n_repeats)
        iterator = batched_dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        # Suppose now we feed in image with lbl0 = '0',
        # and only run experiment on maximizing one specific 
        # dimension of capsule '0'.
        num_class_loop = specs['num_classes'] 
        for i in range(max_epochs): # instance number 
            for j in range(num_class_loop): # j is the index of the target label capsule
                for k in range(n_repeats): # 16 dimensional wise loop
                    try:
                        # Get batched values
                        batch_val = sess.run(batch_data)

                        # Run gradient ascent {iter_n} iterations with step_size={step}
                        # and threshold to get gradient ascended stacked image tensor
                        # (iter_n, 1, 24, 24) and (iter_n, 3, 24, 24)
                        img0 = batch_val['images']
                        iter_n_recorded, ga_img_list = utils.run_gradient_ascent(
                            result_grads[j*num_class_loop+k], img0, batched_images, sess, iter_n, step, threshold)
                        
                        pred_class_prob_list = [] # list of (predicted_class, probabilities of predicted class)s

                        for img in ga_img_list:
                            pred = sess.run(caps_norms_tensor, feed_dict={batched_images: img}) # (1, 10)
                            pred = np.reshape(pred, -1) # (10,)

                            pred_class_prob_list.append(pred)
                        
                        ga_iter_matr = np.array(iter_n_recorded)
                        ga_img_matr = np.stack(ga_img_list, axis=0)
                        pred_class_prob_matr = np.stack(pred_class_prob_list)

                        # save to npz file
                        npzfname = 'instance_{}-cap_{}-dim_{}.npz'.format(i, j, k)
                        npzfname = os.path.join(write_dir, npzfname)
                        np.savez(npzfname, iters=ga_iter_matr, images=ga_img_matr, pred=pred_class_prob_matr)

                        print('{0} {1} total:class:gradient = {2:.1f}% ~ {3:.1f}% ~ {4:.1f}%'.format(
                            ' '*5, '-'*5, 
                            100.0*(i * num_class_loop * n_repeats + j * n_repeats + k + 1) / (max_epochs * num_class_loop * n_repeats),
                            100.0*(j * n_repeats + k + 1)/(num_class_loop * n_repeats),
                            100.0*(k + 1)/n_repeats), end='\r')
                    except tf.errors.OutOfRangeError:
                        break
        print()
        
def explore_direction_aspect(num_gpus, data_dir, dataset, cropped_size,
                             total_batch_size, summary_dir, max_epochs,
                             iter_n, step, threshold, aspect_type):
    """Start direction aspect exploration. Producing results to summary_dir.

    Args:
        num_gpus: number of GPUs available to use.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        max_epochs: maximum epochs to train.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        cropped_size: image size after cropping.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        load_dir: the directory to load files.
        summary_dir: the directory to write files.
        aspect_type: 'naive_max_caps_dim'
    """
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, aspect_type)
    # Declare an empty model graph
    with tf.Graph().as_default():
        # Call run direction aspect
        run_direction_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, cropped_size,
                             iter_n, step, threshold,
                             load_dir, summary_dir, aspect_type)

def run_norm_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, cropped_size,
                    iter_n, step, threshold,
                    load_dir, summary_dir, aspect_type):
    """Start norm aspect exploration. Producing results to summary_dir
    
    Args:
        num_gpus: number of GPUs available to use.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        max_epochs: maximum epochs to train.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        cropped_size: image size after cropping.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        load_dir: the directory to load files.
        summary_dir: the directory to write files.
        aspect_type: 'naive_max_norm' or 'max_norm_diff'
    """
    # Write specs file 
    write_dir = _write_specs_file(summary_dir, aspect_type, dataset, total_batch_size,
                                 max_epochs, iter_n, step, threshold)
    # Find out to feed in noise of data
    if 'noise_' in aspect_type:
        aspect_type = aspect_type[6:]
        split = 'noise'
    else:
        split = 'dream'

    # Find latest checkpoint information
    latest_step, latest_ckpt_path, _ = find_latest_checkpoint_info(load_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph and restore variables
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        saver.restore(sess, latest_ckpt_path)

        # Compute the gradients 
        result_grads, batched_images, caps_norms_tensor = vis_grad_computer[aspect_type].compute_grads(0)
        n_repeats = len(result_grads) # = 10 if it is mnist, fmnist, svhn or cifar10
        print('Number of gradients computed (= n_repeats = number of batches per epoch): ',
              n_repeats)
        
        # Get batched dataset and specs 
        batched_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, cropped_size,
            split=split, n_repeats=n_repeats)
        iterator = batched_dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        num_class_loop = specs['num_classes'] # TODO: =1 when using noise
        for i in range(max_epochs):
            for j in range(num_class_loop):
                for k in range(n_repeats):
                    try:
                        # Get batched values
                        batch_val = sess.run(batch_data)

                        # Run gradient ascent {iter_n} iterations with step_size={step}
                        # and threshold to get gradient ascended stacked image tensor
                        # (iter_n, 1, 24, 24) and (iter_n, 3, 24, 24)
                        img0 = batch_val['images']
                        iter_n_recorded, ga_img_list = utils.run_gradient_ascent(
                            result_grads[k], img0, batched_images, sess, iter_n, step, threshold)
                        
                        pred_class_prob_list = [] # list of probabilities of classes

                        for img in ga_img_list:
                            pred = sess.run(caps_norms_tensor, feed_dict={batched_images: img}) # (1, 10)
                            pred = np.reshape(pred, -1) # (10,)
                            
                            pred_class_prob_list.append(pred) # [(10,), (10,), ...]
                        
                        ga_iter_matr = np.array(iter_n_recorded)
                        ga_img_matr = np.stack(ga_img_list, axis=0)
                        pred_class_prob_matr = np.stack(pred_class_prob_list)

                        # save to npz file
                        npzfname = 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(i, j, k)
                        npzfname = os.path.join(write_dir, npzfname)
                        np.savez(npzfname, iters=ga_iter_matr, images=ga_img_matr, pred=pred_class_prob_matr)

                        print('{0} {1} total:class:gradient = {2:.1f}% ~ {3:.1f}% ~ {4:.1f}%'.format(
                            ' '*5, '-'*5, 
                            100.0*(i * num_class_loop * n_repeats + j * n_repeats + k + 1) / (max_epochs * num_class_loop * n_repeats),
                            100.0*(j * n_repeats + k + 1)/(num_class_loop * n_repeats),
                            100.0*(k + 1)/n_repeats), end='\r')
                    except tf.errors.OutOfRangeError:
                        break
        print()

def explore_norm_aspect(num_gpus, data_dir, dataset, cropped_size,
                        total_batch_size, summary_dir, max_epochs,
                        iter_n, step, threshold, aspect_type):
    """Produce gradient ascent on given images.

    Args:
        num_gpus: number of GPUs available to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        cropped_size: image size after cropping.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        summary_dir: the directory to write files.
        max_epochs: maximum epochs to train.
        iter_n: number of iterations to add gradients to original image.
        step: step size of each iteration of gradient ascent to mutliply.
        threshold: any gradients less than this value will not be added to the original image.
        aspect_type: 'naive_max_norm' or 'max_norm_diff', or 'noise_naive_max_norm' or 'noise_max_norm_diff'
    """
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, aspect_type)
    # Declare an empty model graph
    with tf.Graph().as_default():
        # Call runn norm aspect
        run_norm_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, cropped_size,
                        iter_n, step, threshold,
                        load_dir, summary_dir, aspect_type)

def run_evaluate_session(iterator, specs, load_dir, summary_dir, kind):
    """Find available checkpoints and iteratively load the graph and variables.

    Args:
        iterator: iterator, dataset iterator.
        specs: dict, dictionary containing dataset specifications.
        load_dir: str, directory that contains checkpoints.
        summary_dir: str, directory to write summary
        kind: 'train' or 'test'
    Raises:
        ckpt files not found.
    """
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # section to write test_history
    """Load available checkpoints"""
    latest_step, latest_ckpt_path, all_step_ckpt_pairs = find_latest_checkpoint_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        print('Found ckpt at step {}'.format(latest_step))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    
    with open(os.path.join(summary_dir, '%s_history.txt' % kind), 'a+') as f:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Import compute grah
            saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
            batch_data = iterator.get_next()

            # Iteratively restore variables
            for idx, (step, ckptpath) in enumerate(all_step_ckpt_pairs):
                # Restore variables 
                saver.restore(sess, ckptpath)
                
                sess.run(iterator.initializer)
                accs = []

                while True:
                    try: 
                        # Get placeholders and create feed dict
                        feed_dict = {}
                        for i in range(specs['num_gpus']):
                            batch_val = sess.run(batch_data)
                            feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                            feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']

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
                mean_acc = np.mean(accs)
                print('step: {0}, accuracy: {1:.4f} ~ {2} / {3}'.format(step, mean_acc, idx + 1, len(all_step_ckpt_pairs)))
                f.write('{}, {}\n'.format(step, mean_acc))

def evaluate(num_gpus, data_dir, dataset, model_type, total_batch_size, cropped_size,
             summary_dir, max_epochs):
    """Iteratively restore the graph and variables, and return the data to 
    train and test curve.

    Args:
        num_gpus: number of GPUs available to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        model_type: the name of the model architecture.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        cropped_size: image size after cropping.
        summary_dir: the directory to load the model.
        max_epochs: maximum epochs to evaluate, ≡ 1.
    """
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, 'evaluate')
    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get train batched dataset and declare initializable iterator
        train_distributed_dataset, train_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, cropped_size,
            'train')
        train_iterator = train_distributed_dataset.make_initializable_iterator()
        # Call evaluate experiment 
        run_evaluate_session(train_iterator, train_specs, load_dir, summary_dir, 'train')
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        test_distributed_dataset, test_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
             data_dir, dataset, cropped_size,
             'test')
        test_iterator = test_distributed_dataset.make_initializable_iterator()
        # Call evaluate experiment
        run_evaluate_session(test_iterator, test_specs, load_dir, summary_dir, 'test')

def run_glitch_session(iterator, specs, load_dir, summary_dir, kind):
    """Find available checkpoints run predictions
    
    Args:
        iterator: iterator, dataset iterator;
        specs: dict, dictionary containing dataset specifications;
        load_dir: str, directory that contains checkpoints;
        summary_dir: str, directory to write summary;
        kind: 'train' or 'test';
    Raises:
        ckpts not found
    """
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # section to write error predictions
    """Load available checkpoints"""
    latest_step, latest_ckpt_path, _ = find_latest_checkpoint_info(load_dir, False)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('Checkpoint files not found!')
    else:
        print('Found ckpt at step {}'.format(latest_step))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Import compute graph and restore variables
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        saver.restore(sess, latest_ckpt_path)

        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        logits10_ts = []
        batched_images_ts = []
        batched_labels_ts = []
        for i in range(specs['num_gpus']):
            logits10_ts.append(tf.get_collection('tower_%d_visual' % i)[-1])
            batched_images_ts.append(tf.get_collection('tower_%d_batched_images' % i)[0])
            batched_labels_ts.append(tf.get_collection('tower_%d_batched_labels' % i)[0])
        
        logits10_t = tf.concat(logits10_ts, 0)
        batched_images_t = tf.concat(batched_images_ts, 0)
        batched_labels_t = tf.concat(batched_labels_ts, 0)

        logits_t = tf.argmax(logits10_t, axis=1, output_type=tf.int32)
        labels_t = tf.argmax(batched_labels_t, axis=1, output_type=tf.int32)

        correct_t = tf.equal(logits_t, labels_t)
        error_t = tf.logical_not(correct_t)

        # get accuracy
        res_acc = tf.get_collection('accuracy')[0]

        while True:
            try:
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    feed_dict[batched_images_ts[i]] = batch_val['images']
                    feed_dict[batched_labels_ts[i]] = batch_val['labels']

                accuracy, logits10, labels, error = sess.run([res_acc, logits10_t, batched_labels_t, error_t], feed_dict=feed_dict)

                # wrongly predicted instances
                error_indices = [i for i in range(len(error)) if error[i] == True]
                print(labels[error_indices])
                print(logits10[error_indices])

            except tf.errors.OutOfRangeError:
                break

def glitch(num_gpus, data_dir, dataset, model_type, total_batch_size, cropped_size, 
           summary_dir, max_epochs):
    """Predict all the error predicted instances and store them into writen files.

    Args:
        num_gpus: number of GPUs to use.
        data_dir: the directory containing the input data.
        dataset: the name of the dataset for the experiments.
        model_type: the name of the model architecture.
        total_batch_size: total batch size, will be distributed to `num_gpus` GPUs.
        cropped_size: image size after cropping.
        summary_dir: the directory to write summaries and save the model.
        max_epochs: maximum epochs to train.
    """
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, 'glitch')
    # Declare an empty model graph
    with tf.Graph().as_default():
        # Get train batched dataset and declare initializable iterator
        train_distributed_dataset, train_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, cropped_size,
            'train')
        train_iterator = train_distributed_dataset.make_initializable_iterator()
        # Call evaluate experiment 
        run_glitch_session(train_iterator, train_specs, load_dir, summary_dir, 'train')
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        test_distributed_dataset, test_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
             data_dir, dataset, cropped_size,
             'test')
        test_iterator = test_distributed_dataset.make_initializable_iterator()
        # Call evaluate experiment
        run_glitch_session(test_iterator, test_specs, load_dir, summary_dir, 'test')

def run_train_session(iterator, specs, # Dataset related
                      summary_dir, max_to_keep, max_epochs, # Checkpoint related
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
        epochs_done = 0
        # restore ckpt if not restart
        latest_step, latest_checkpoint_fpath, _ = find_latest_checkpoint_info(summary_dir, False)
        if latest_step != -1 and latest_checkpoint_fpath != None:
            saver.restore(sess, latest_checkpoint_fpath)
            step_counter = latest_step
            epochs_done = step_counter // specs['steps_per_epoch']
        total_steps = specs['steps_per_epoch'] * (max_epochs - epochs_done)

        # Start feeding process
        for _ in range(total_steps):
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

def train(hparams, num_gpus, data_dir, dataset, model_type, total_batch_size, cropped_size,
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
        cropped_size: image size after cropping.
        summary_dir: the directory to write summaries and save the model.
        max_to_keep: maximum checkpoint files to keep.
        save_epochs: how often the training model should be saved.
        max_epochs: maximum epochs to train.
    """
    summary_dir = os.path.join(summary_dir, 'train')

    # Declare the empty model graph
    with tf.Graph().as_default():
        # Get batched dataset and declare initializable iterator
        distributed_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, cropped_size,
            'train')
        iterator = distributed_dataset.make_initializable_iterator()
        # Initialize model with hparams and specs
        model = models[model_type](hparams, specs)
        # Build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        joined_result = model.build_model_on_multi_gpus()

        """Print stats"""
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        """"""
        # Clear summary directory, TODO: start train from where left.
        # Call train experiment

        run_train_session(iterator, specs,
                          summary_dir, max_to_keep, max_epochs,
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
        train(hparams, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size,
                       FLAGS.summary_dir, FLAGS.max_to_keep,
                       FLAGS.save_epochs, FLAGS.max_epochs)
    elif FLAGS.mode == 'glitch':
        glitch(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size, 
              FLAGS.summary_dir, FLAGS.max_epochs)
    elif FLAGS.mode == 'evaluate':
        evaluate(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size,
                 FLAGS.summary_dir, FLAGS.max_epochs)
    elif FLAGS.mode in NORM_ASPECT_TYPES or FLAGS.mode in ['noise_' + aspect for aspect in NORM_ASPECT_TYPES]:
        explore_norm_aspect(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.image_size,
                            FLAGS.total_batch_size, FLAGS.summary_dir, FLAGS.max_epochs,
                            FLAGS.iter_n, float(FLAGS.step), float(FLAGS.threshold), FLAGS.mode)
    elif FLAGS.mode in DIRECTION_ASPECT_TYPES or FLAGS.mode in ['noise_' + aspect for aspect in DIRECTION_ASPECT_TYPES]:
        explore_direction_aspect(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.image_size,
                                 FLAGS.total_batch_size, FLAGS.summary_dir, FLAGS.max_epochs, 
                                 FLAGS.iter_n, float(FLAGS.step), float(FLAGS.threshold), FLAGS.mode)
    else:
        raise ValueError(
            "No matching mode found for '{}'".format(FLAGS.mode))
    
if __name__ == '__main__':
    tf.app.run()
