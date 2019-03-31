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

"""Framework for
1. train
2. evaluate (compare capsule norms)
3. explore norm aspect
4. explore direction aspect
5. ensemble evaluate (ensemble results from deleted capsule norms)
"""

from __future__ import absolute_import, division, print_function

import os 
import sys 
import time 
import re 
from glob import glob
from pprint import pprint
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

from config import FLAGS, default_hparams

MODELS = {
    'cnn': cnn_model.CNNModel,
    'cap': capsule_model.CapsuleModel
}

INPUTS = {
    'mnist': mnist_input,
    'fashion_mnist': fashion_mnist_input,
    'svhn': svhn_input,
    'cifar10': cifar10_input
}

DREAM_INPUTS = {
    'mnist': mnist_dream_inputs,
    'fashion_mnist': fashion_mnist_dream_input,
    'svhn': svhn_dream_input,
    'cifar10': cifar10_dream_input
}

VIS_GRAD_COMPUTER = {
    'naive_max_norm': naive_max_norm,
    'max_norm_diff': max_norm_diff,
    'naive_max_caps_dim': naive_max_caps_dim,
    'max_caps_dim_diff': max_caps_dim_diff
}

METHOD_TYPES = ['normal', 'ensemble']

NORM_ASPECT_TYPES = ['naive_max_norm', 'max_norm_diff']

DIRECTION_ASPECT_TYPES = ['naive_max_caps_dim', 'max_caps_dim_diff']

def _compute_entropy(arr):
    """Given a numpy array compute the entropy of it
    Args:
        arr: a numpy array;
    Returns:
        entropy: scalar, the entropy of the given array;
    """
    arr_sum = np.sum(arr)
    arr_exp = np.exp(arr)
    entropy = - np.dot(arr_exp/arr_sum, arr)
    return entropy

def get_distributed_dataset(total_batch_size, num_gpus,
                            max_epochs, data_dir, dataset, image_size,
                            split='default', n_repeats=None):
    """Reads the input data using 'input_data' functions.

    For 'train' and 'test' splits,
        given {num_gpus} GPUs and {total_batch_size}, we distribute
        those {total_batch_size} into {num_gpus} partitions,
        denoted as {batch_size}.

    For 'noise' and 'dream' splits,
        check if {total_batch_size} ≡ 1, otherwise raise 'ValueError'.
        In this case, we will duplicate every example {num_gpus} times 
        so that when we pass the examples into multi-tower models, it is 
        calculating and averaging the gradients of the same images.

    Args:
        total_batch_size: total number of data entries over all towers;
        num_gpus: number of GPUs available to use;
        max_epochs: for 'train' split, this parameter decides the number of 
            epochs to train for the model; for 'test' split, this parameter
            should ≡ 1.
        data_dir: the directory containing the data;
        dataset: the name of the dataset;
        image_size: image size after cropping;
        split: 'train', 'test', 'noise', 'dream';
        n_repeats ('noise' and 'dream'): the number of repeats of the same image.
    Returns:
        batched_dataset: dataset object;
        specs: dataset specifications.
    """
    assert dataset in ['mnist', 'fashion_mnist', 'svhn', 'cifar10']
    with tf.device('/gpu:0'):
        if split in ['train', 'test']:
            assert total_batch_size % num_gpus == 0
            distributed_dataset, specs = INPUTS[dataset].inputs(
                total_batch_size, num_gpus, max_epochs, image_size, 
                data_dir, split)
            return distributed_dataset, specs
        elif split == 'noise':
            batched_dataset, specs = noise_dream_input.inputs(
                'noise', 1, max_epochs, n_repeats, image_size)
            return batched_dataset, specs
        elif split == 'dream':
            batched_dataset, specs = DREAM_INPUTS[dataset].inputs(
                'train', data_dir, max_epochs, n_repeats, image_size)
            return batched_dataset, specs
        else:
            raise ValueError()

def find_event_file_path(load_dir):
    """Finds the event file.

    Args:
        load_dir: the directory to look for the training checkpoints.
    Returns:
        path to the event file.
    """
    fpath_list = glob(os.path.join(load_dir, 'events.*'))
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
            ckpt_paths = glob(os.path.join(load_dir, 'model.ckpt-*.index'))
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

def run_train_session(iterator, specs, 
                      summary_dir, max_epochs,
                      joined_result, save_epochs):
    """Starts a session, train the model, write summary into an event file,
    and save the whole graph one time and variable every {save_epochs} epochs.
    
    Args:
        iterator: dataset iterator;
        specs: dict, dataset specifications;
        summary_dir: str, directory to store ckpts;
        joined_result: namedtuple, TowerResult('inferred', 'train_op',
                                               'summary', 'correct', 'accuracy');
        save_epochs: scalar, how often to save the data.
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare summary writer and save the graph in the meanwhile
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        # declar batched data instance and initialize the iterator
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)
        # initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # declare saver object for future saving
        saver = tf.train.Saver(max_to_keep=None)

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

        # start feeding process
        for _ in range(total_steps):
            start_anchor = time.time() # time anchor
            step_counter += 1

            try:
                # get placeholders and create feed_dict
                feed_dict = {} 
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                    feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                
                """Run inferences"""
                summary, accuracy, _ = sess.run(
                    [joined_result.summary, joined_result.accuracy, joined_result.train_op],
                    feed_dict=feed_dict)
                """Add summary"""
                writer.add_summary(summary, global_step=step_counter)
                # calculate time
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

def train(hparams, num_gpus, data_dir, dataset, model_type, total_batch_size, image_size,
                   summary_dir, save_epochs, max_epochs):
    """Trains a model.

    It will initialize the model with either previously a saved model ckpt in
    the {summary_dir} directory or start from scratch if the directory is empty.
    The training is distributed on {num_gpus} GPUs. It writes a summary at 
    every step and saves the model every {save_epochs} epochs.

    Args:
        hparams: the hyperparameters to build the model graph;
        num_gpus: number of GPUs to use;
        data_dir: the directory containing the input data;
        dataset: the name of the dataset for the experiment;
        model_type: the name of model architecture;
        total_batch_size: total batch size, which will be distributed to {num_gpus} GPUs;
        image_size: image size after cropping/resizing;
        summary_dir: the directory to write summaries and save the model;
        save_epochs: how often the training model should be saved;
        max_epochs: maximum epochs to train.
    """
    # define subfolder in {summary_dir}
    summary_dir = os.path.join(summary_dir, 'train')
    # define model graph
    with tf.Graph().as_default():
        # get batched dataset and declare initializable iterator
        distributed_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
            data_dir, dataset, image_size,
            'train')
        iterator = distributed_dataset.make_initializable_iterator()
        # initialize model with hparams and specs
        model = MODELS[model_type](hparams, specs)
        # build a model on multiple gpus and returns a tuple of 
        # (a list of input tensor placeholders, a list of output tensor placeholders)
        joined_result = model.build_model_on_multi_gpus()

        """Print stats"""
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        """"""

        run_train_session(iterator, specs, 
                          summary_dir, max_epochs,
                          joined_result, save_epochs)

def run_evaluate_session(iterator, specs, load_dir, summary_dir, kind, 
                         model_type, threshold):
    """Find available ckpts and iteratively load the graph and variables.

    Args:
        iterator: dataset iterator;
        specs: dict, dataset specifications;
        load_dir: str, directory to load graph;
        summary_dir: str, directory to store ckpts;
        kind: 'train' or 'test';
        model_type: 'cnn' or 'cap';
        threhold: if {model_type}='cnn', then it should be None; 
            else, it is threshold to filter capsules;
    """
    # create summary folder if not exists
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    """Load available ckpts"""
    # find latest step, ckpt, and all step-ckpt pairs
    latest_step, latest_ckpt_path, all_step_ckpt_pairs = find_latest_checkpoint_info(load_dir, True)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not fould!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # get dataset object working
        batch_data = iterator.get_next()

        """Process ensemble evaluation tensors"""
        if model_type == 'cap':
            # print ensemble capsule tensors
            for i in range(specs['num_gpus']):
                pprint(tf.get_collection('tower_%d_ensemble_acts' % i))
            ensemble_votes_list = []
            batched_images_t_list = []
            batched_labels_t_list = []
            for i in range(specs['num_gpus']):
                ensemble_votes_list.append(tf.get_collection('tower_%d_ensemble_acts' % i))
                batched_images_t_list.append(tf.get_collection('tower_%d_batched_images' % i)[0])
                batched_labels_t_list.append(tf.get_collection('tower_%d_batched_labels' % i)[0])
            removed_effect_votes = []
            for i in range(specs['num_classes']):
                removed_effect_votes.append(
                    tf.concat([ensemble_votes_list[j][i] for j in range(specs['num_gpus'])], 0))
            batched_images_t = tf.concat(batched_images_t_list, 0)
            batched_labels_t = tf.concat(batched_labels_t_list, 0)
            batched_ensemble_logits10_t = tf.add_n(removed_effect_votes)

            images_t = batched_images_t
            labels_t = tf.argmax(batched_labels_t, axis=1, output_type=tf.int32)
            logits_t = tf.argmax(batched_ensemble_logits10_t, axis=1, output_type=tf.int32)

            correct_t = tf.cast(tf.equal(logits_t, labels_t), tf.float32)

            ensemble_acc_t = tf.reduce_mean(correct_t)

        acc_t = tf.get_collection('accuracy')[0]

        if model_type == 'cap':
            step_mean_ensemble_acc_pairs = []
        step_mean_acc_pairs = []

        # iteratively restore variables and run evaluations
        for idx, (step, ckptpath) in enumerate(all_step_ckpt_pairs):
            # restore variables
            saver.restore(sess, ckptpath)

            sess.run(iterator.initializer)

            if model_type == 'cap':
                ensemble_accs = []
            accs = []

            while True:
                try: 
                    # get placeholders and create feed dict
                    feed_dict = {}
                    for i in range(specs['num_gpus']):
                        batch_val = sess.run(batch_data)
                        feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                        feed_dict[tf.get_collection('tower_%d_batched_labels' % i)[0]] = batch_val['labels']
                        if model_type == 'cap':
                            feed_dict[tf.get_collection('tower_%d_batched_threshold' % i)[0]] = threshold
                    
                    if model_type == 'cap':
                        ensemble_acc, acc = sess.run([ensemble_acc_t, acc_t], feed_dict=feed_dict)
                        ensemble_accs.append(ensemble_acc)
                    else:
                        acc = sess.run(acc_t, feed_dict=feed_dict)
                    accs.append(acc)
                except tf.errors.OutOfRangeError:
                    break
            if model_type == 'cap':
                mean_ensemble_acc = np.mean(ensemble_accs)
                step_mean_ensemble_acc_pairs.append((step, mean_ensemble_acc))
            mean_acc = np.mean(accs)
            step_mean_acc_pairs.append((step, mean_acc))
            if model_type == 'cap':
                print('step: {0}, accuracy:ensemble = {1:.4f}: {2:.4f} ~ {3} / {4}'.format(step, mean_acc, mean_ensemble_acc, idx+1, len(all_step_ckpt_pairs)))
            else:
                print('step: {0}, accuracy = {1:.4f} ~ {2} / {3}'.format(step, mean_acc, idx+1, len(all_step_ckpt_pairs)))
        if model_type == 'cap':
            with open(os.path.join(summary_dir, '%s_ensemble_history.txt' % kind), 'w+') as f:
                for step, mean_ensemble_acc in step_mean_ensemble_acc_pairs:
                    f.write('{}, {}\n'.format(step, mean_ensemble_acc))
        with open(os.path.join(summary_dir, '%s_history.txt') % kind, 'w+') as f:
            for step, mean_acc in step_mean_acc_pairs:
                f.write('{}, {}\n'.format(step, mean_acc))

def evaluate(num_gpus, data_dir, dataset, model_type, total_batch_size, image_size,
             threshold, summary_dir, max_epochs):
    """Iteratively restore the graph and variables, and return the data to train and test curve.
    
    Args:
        num_gpus: number of GPUs to use;
        data_dir: the directory containing the input data;
        dataset: the name of the dataset for the experiment;
        model_type: the name of model architecture;
        total_batch_size: total batch size, which will be distributed to {num_gpus} GPUs;
        image_size: image size after cropping/resizing;
        threshold: threshold to filter out the target capsule effect;
        summary_dir: the directory to write summaries and save the model;
        max_epochs: maximum epochs to evaluate, ≡ 1.
    """
    # define subfolder to load ckpt and write related files
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, 'evaluate')
    # declare an empty model graph
    with tf.Graph().as_default():
        # get train batched dataset and declare initializable iterator
        train_distributed_dataset, train_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
            data_dir, dataset, image_size,
            'train')
        train_iterator = train_distributed_dataset.make_initializable_iterator()
        # call evaluate experiment
        run_evaluate_session(train_iterator, train_specs, load_dir, summary_dir, 'train', 
                             model_type, threshold)
    with tf.Graph().as_default():
        # get test batched dataset and delcare initializable iterator
        test_distributed_dataset, test_specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
            data_dir, dataset, image_size,
            'test')
        test_iterator = test_distributed_dataset.make_initializable_iterator()
        # call evaluate experiment
        run_evaluate_session(test_iterator, test_specs, load_dir, summary_dir, 'test', 
                             model_type, threshold)

def run_test_session(iterator, specs, load_dir):
    """Load available ckpts"""
    latest_step, latest_ckpt_path, _ = find_latest_checkpoint_info(load_dir, False)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\n ckpt files not found!\n {0}'.format('='*20))
    else:
        print('{0}\nFound a ckpt!\n{0}'.format('='*20))
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # import compute graph
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        # get dataset object working
        batch_data = iterator.get_next()

        acc_t = tf.get_collection('accuracy')[0]
        
        # restore variables 
        saver.restore(sess, latest_ckpt_path)
        sess.run(iterator.initializer)

        accs = []
        while True:
            try:
                feed_dict = {}
                for i in range(specs['num_gpus']):
                    batch_val = sess.run(batch_data)
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['images']
                    feed_dict[tf.get_collection('tower_%d_batched_images' % i)[0]] = batch_val['labels']
                acc = sess.run(acc_t, feed_dict=feed_dict)
                accs.append(acc)
            except tf.errors.OutOfRangeError:
                break
        mean_acc = np.mean(accs)
        print(mean_acc)

def test(split, num_gpus, data_dir, dataset, total_batch_size, image_size, max_epochs):
    # define subfolder to load ckpt
    load_dir = os.path.join(summary_dir, 'train')
    # declare an empty model graph
    with tf.Graph().as_default():
        # get train batched dataset and declare initializable iterator
        distributed_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
            data_dir, dataset, image_size,
            split)
        iterator = distributed_dataset.make_initializable_iterator()
        # call test experiment
        run_test_session(iterator, specs, load_dir)

def run_norm_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, image_size,
                    iter_n, step, threshold,
                    load_dir, summary_dir, aspect_type):
    """Run norm aspect exploration. Producing results to summary_dir.
    
    Args:
        num_gpus: number of GPUs available to use;
        total_batch_size: total batch size, ≡ 1;
        max_epochs: maximum epochs to train;
        data_dir: the directory containing the input data;
        dataset: the name of the dataset for the experiments;
        image_size: image size after cropping or resizing;
        iter_n: number of iterations to add gradients to original image;
        step: step size of each iteration of gradient ascent to mutliply;
        threshold: any gradients less than this value will not be added to the original image;
        load_dir: the directory to load files;
        summary_dir: the directory to write files;
        aspect_type: 'naive_max_norm' or 'max_norm_diff'.
    """
    # wrtie specs file 
    write_dir = _write_specs_file(summary_dir, aspect_type, dataset, total_batch_size,
                                  max_epochs, iter_n, step, threshold)
    # find out whether to feed in noise or normal images
    if 'noise_' in aspect_type:
        aspect_type = aspect_type[6:]
        split = 'noise'
    else:
        split = 'dream'

    # find latest ckpt information
    latest_step, latest_ckpt_path, _ = find_latest_checkpoint_info(load_dir)
    if latest_step == -1 or latest_ckpt_path == None:
        raise ValueError('{0}\nckpt files not found!\n{0}'.format('='*20))
    else:
        latest_ckpt_meta_path = latest_ckpt_path + '.meta'
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # import compute graph and restore variables 
        saver = tf.train.import_meta_graph(latest_ckpt_meta_path)
        saver.restore(sess, latest_ckpt_path)

        # compute the gradients
        result_grads, batched_images, caps_norms_tensor = VIS_GRAD_COMPUTER[aspect_type].compute_grads(0)
        n_repeats = len(result_grads)
        print('Number of gradients computed (= n_repeats = number of batches per epoch): ',
              n_repeats)
        
        batched_labels_t = tf.get_collection('tower_%d_batched_labels' % 0)[0]

        # get batched dataset and specs
        batched_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs,
            data_dir, dataset, image_size,
            split=split, n_repeats=n_repeats)
        iterator = batched_dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
        sess.run(iterator.initializer)

        if split == 'noise':
            num_class_loop = 1
        else:
            num_class_loop = specs['num_classes'] 
        for i in range(max_epochs):
            for j in range(num_class_loop):
                for k in range(n_repeats):
                    try:
                        # get batched values
                        batch_val = sess.run(batch_data)

                        # run gradient ascent {iter_n} iterations with {step} step size
                        # and threshold to get gradient ascended stacked image tensor
                        # (iter_n, 1, 24, 24) and (iter_n, 3, 24, 24)
                        img0 = batch_val['images']
                        iter_n_recorded, ga_img_list = utils.run_gradient_ascent(
                            result_grads[k], img0, batched_images, sess, iter_n, step, threshold)
                        
                        pred_class_prob_list = [] # list of probabilities of classes
                        pred_class_entropy_list = [] # list of probabilities of prediction entropies
                        for img in ga_img_list:
                            pred = sess.run(caps_norms_tensor, feed_dict={batched_images: img}) # (1, 10)
                            pred = np.reshape(pred, -1) # (10,)
                            pred_cl = np.argmax(pred) # ()

                            entropy = _compute_entropy(pred)
                            
                            # winning capsule mask
                            win_cap_mask = np.array([0.0 for _ in range(10)])
                            win_cap_mask[pred_cl] = 1.0
                            win_cap_mask = np.expand_dims(win_cap_mask, axis=0)
                            # all capsules mask
                            all_cap_mask = np.expand_dims(np.array([1.0 for _ in range(10)]), axis=0)

                            pred_class_prob_list.append(pred) # [(10,), (10,), ...]
                            pred_class_entropy_list.append(entropy)

                        ga_iter_matr = np.array(iter_n_recorded)
                        ga_img_matr = np.stack(ga_img_list, axis=0)
                        pred_class_prob_matr = np.stack(pred_class_prob_list)
                        pred_class_entropy_matr = np.stack(pred_class_entropy_list, axis=0)

                        # save to npz file
                        npzfname = 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(i, j, k)
                        npzfname = os.path.join(write_dir, npzfname)
                        np.savez(npzfname, iters=ga_iter_matr, images=ga_img_matr, pred=pred_class_prob_matr, 
                                 pred_entropy=pred_class_entropy_matr)

                        print('{0} {1} total:class:gradient = {2:.1f}% ~ {3:.1f}% ~ {4:.1f}%'.format(
                            ' '*5, '-'*5, 
                            100.0*(i * num_class_loop * n_repeats + j * n_repeats + k + 1) / (max_epochs * num_class_loop * n_repeats),
                            100.0*(j * n_repeats + k + 1)/(num_class_loop * n_repeats),
                            100.0*(k + 1)/n_repeats), end='\r')
                    except tf.errors.OutOfRangeError:
                        break
        print()

def explore_norm_aspect(num_gpus, data_dir, dataset, image_size,
                        total_batch_size, summary_dir, max_epochs,
                        iter_n, step, threshold,
                        aspect_type):
    """Run gradient ascent on given images.
    
    Args:
        num_gpus: number of GPUs available to use;
        data_dir: the directory containing the input data;
        dataset: dataset name;
        image_size: image size after cropping or resizing;
        total_batch_size: total batch size, ≡ 1;
        summary_dir: the directory to write files;
        max_epochs: numbers of instance to use;
        iter_n: number of iterations to add gradients to original images;
        step: step size of each iteration of gradient ascent;
        threshold: any gradients less than this value will not be added to the original images;
        aspect_type: 'naive_max_norm', 'max_norm_diff', or 'noise_naive_max_norm', 'noise_max_norm_diff'.
    """
    # define load_dir and summary_dir
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, aspect_type)
    # declare an empty model graph
    with tf.Graph().as_default():
        # call run_norm_aspect
        run_norm_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, image_size,
                        iter_n, step, threshold,
                        load_dir, summary_dir, aspect_type)

def run_direction_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, image_size,
                         iter_n, step, threshold,
                         load_dir, summary_dir, aspect_type):
    """Run direction aspect exploration. Producing results to summary_dir.

    Args:
        num_gpus: number of GPUs available to use;
        total_batch_size: total batch size. ≡ 1;
        max_epochs: number of instance to produce;
        data_dir: the directory containing the input data;
        dataset: dataset name;
        image_size: image size after cropping or resizing;
        iter_n: number of iterations to add gradients to original images;
        step: step size of each iteration of gradient ascent to multiply;
        threshold: any gradients less than this value will not be added to original images;
        load_dir: the directory to load files;
        summary_dir: the directory to write files;
        aspect_type: 'naive_max_caps_dim', 'max_caps_dim_diff', or 'noise_naive_max_caps_dim', 'max_caps_dim_diff'.
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
        result_grads, batched_images, caps_norms_tensor= VIS_GRAD_COMPUTER[aspect_type].compute_grads(0)
        n_repeats = 16 # 16 dimensional vector
        print('Number of gradients computed: ', len(result_grads))

        batched_labels_t = tf.get_collection('tower_%d_batched_labels' % 0)[0]

        # Get batched dataset and specs
        batched_dataset, specs = get_distributed_dataset(
            total_batch_size, num_gpus, max_epochs, 
            data_dir, dataset, image_size,
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
                        pred_class_entropy_list = []

                        for img in ga_img_list:
                            pred = sess.run(caps_norms_tensor, feed_dict={batched_images: img}) # (1, 10)
                            pred = np.reshape(pred, -1) # (10,)
                            pred_cl = np.argmax(pred) # ()

                            entropy = _compute_entropy(pred)

                            # winning capsule mask
                            win_cap_mask = np.array([0.0 for _ in range(10)])
                            win_cap_mask[pred_cl] = 1.0 
                            win_cap_mask = np.expand_dims(win_cap_mask, axis=0)
                            # all capsules mask
                            all_cap_mask = np.expand_dims(np.array([1.0 for _ in range(10)]), axis=0)

                            pred_class_prob_list.append(pred)
                            pred_class_entropy_list.append(entropy)

                        ga_iter_matr = np.array(iter_n_recorded)
                        ga_img_matr = np.stack(ga_img_list, axis=0)
                        pred_class_prob_matr = np.stack(pred_class_prob_list)
                        pred_class_entropy_matr = np.stack(pred_class_entropy_list, axis=0)

                        # save to npz file
                        npzfname = 'instance_{}-cap_{}-dim_{}.npz'.format(i, j, k)
                        npzfname = os.path.join(write_dir, npzfname)
                        np.savez(npzfname, iters=ga_iter_matr, images=ga_img_matr, pred=pred_class_prob_matr,
                                 pred_entropy=pred_class_entropy_matr)

                        print('{0} {1} total:class:gradient = {2:.1f}% ~ {3:.1f}% ~ {4:.1f}%'.format(
                            ' '*5, '-'*5, 
                            100.0*(i * num_class_loop * n_repeats + j * n_repeats + k + 1) / (max_epochs * num_class_loop * n_repeats),
                            100.0*(j * n_repeats + k + 1)/(num_class_loop * n_repeats),
                            100.0*(k + 1)/n_repeats), end='\r')
                    except tf.errors.OutOfRangeError:
                        break
        print()

def explore_direction_aspect(num_gpus, data_dir, dataset, image_size,
                             total_batch_size, summary_dir, max_epochs,
                             iter_n, step, threshold, aspect_type):
    """Start direction aspect exploration. Producing results to summary_dir.

    Args:
        num_gpus: number of GPUs available to use;
        data_dir: the directory containing the input data;
        dataset: the name of the dataset for the experiments;
        image_size: image size after cropping or resizing;
        total_batch_size: total batch size, ≡ 1;
        summary_dir: the directory to write files;
        max_epochs: number of different instance for the same class;
        iter_n: number of iterations to add gradients to original image;
        step: step size of each iteration of gradient ascent to mutliply;
        threshold: any gradients less than this value will not be added to the original image;
        aspect_type: 'naive_max_caps_dim', 'max_caps_dim_diff', or 'noise_naive_max_caps_dim', 'max_caps_dim_diff'.
    """
    # define load_dir and summary_dir
    load_dir = os.path.join(summary_dir, 'train')
    summary_dir = os.path.join(summary_dir, aspect_type)
    # delare an empty model graph
    with tf.Graph().as_default():
        # call run direction aspect
        run_direction_aspect(num_gpus, total_batch_size, max_epochs, data_dir, dataset, image_size,
                             iter_n, step, threshold,
                             load_dir, summary_dir, aspect_type)

def main(_):
    hparams = default_hparams()
    if FLAGS.hparams_override:
        hparams.parse(FLAGS.hparams_override)
    
    if FLAGS.mode == 'train':
        train(hparams, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size, 
                       FLAGS.summary_dir, FLAGS.save_epochs, FLAGS.max_epochs)
    if FLAGS.mode == 'test':
        test(FLAGS.split, FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.total_batch_size, FLAGS.image_size, FLAGS.max_epochs)
    elif FLAGS.mode == 'evaluate':
        evaluate(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.model, FLAGS.total_batch_size, FLAGS.image_size,
                 FLAGS.threshold, FLAGS.summary_dir, FLAGS.max_epochs)
    elif FLAGS.mode == 'glitch':
        pass
    elif FLAGS.mode in NORM_ASPECT_TYPES or FLAGS.mode in ['noise_' + aspect for aspect in NORM_ASPECT_TYPES]:
        explore_norm_aspect(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.image_size,
                            FLAGS.total_batch_size, FLAGS.summary_dir, FLAGS.max_epochs,
                            FLAGS.iter_n, float(FLAGS.step), float(FLAGS.threshold),
                            FLAGS.mode)
    elif FLAGS.mode in DIRECTION_ASPECT_TYPES or FLAGS.mode in ['noise_' + aspect for aspect in DIRECTION_ASPECT_TYPES]:
        explore_direction_aspect(FLAGS.num_gpus, FLAGS.data_dir, FLAGS.dataset, FLAGS.image_size,
                                 FLAGS.total_batch_size, FLAGS.summary_dir, FLAGS.max_epochs,
                                 FLAGS.iter_n, float(FLAGS.step), float(FLAGS.threshold),
                                 FLAGS.mode)
    else:
        raise ValueError("No matching mode found for '{}'".format(FLAGS.mode))

if __name__ == '__main__':
    tf.app.run()
    