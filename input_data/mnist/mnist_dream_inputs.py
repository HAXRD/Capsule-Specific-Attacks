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
import tensorflow as tf 
import numpy as np
import os

def _dream_cropping(image, label, specs, cropped_size):

    image = tf.expand_dims(image, -1) # (HWC)
    if cropped_size < specs['image_size']:
        image = tf.image.resize_image_with_crop_or_pad(
            image, cropped_size, cropped_size)
    
    # convert from 0 ~ 255 to 0. ~ 1.
    image = tf.cast(image, tf.float32) * (1. / 255.)
    # transpose image into (CHW)
    image = tf.transpose(image, [2, 0, 1]) # (HWC)

    feature = {
        'image':image,
        'label': tf.one_hot(label, 10)
    }
    return feature

def _dream_process(feature):

    batched_features = {
        'images': feature['image'],
        'labels': feature['label']
    }
    return batched_features

def _dream_sample_pairs(split, data_dir, max_epochs, n_repeats,
                        seed=123, total_batch_size=1):
    """
    We do the following steps to produce the dataset:
        1. sample one (image, label) pair in one class;
        2. repeat pair in 1. {n_repeats} times;
        3. go back to do 1. unless we finish one iteration 
           (after a {num_classes} time loop). And we consider
           this as one epoch.
        4. go back to do 1. again to finish {max_epochs} loop.
    So there will be {max_epochs} number of unique pairs selected for 
    each class.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the mnist data directory.
        max_epochs: maximum epochs to go through the model.
        n_repeats: number of computed gradients
        seed: seed to produce pseudo randomness.
        batch_size: total number of images per batch.
    Returns:
        processed images, labels and specs
    """
    
    """Dataset specs"""
    specs = {
        'split': split,
        'max_epochs': max_epochs,
        'steps_per_epoch': n_repeats,
        'batch_size': total_batch_size,
        'image_size': 28,
        'depth': 1,
        'num_classes': 10
    }
    """Load data from numpy array file"""
    with np.load(os.path.join(data_dir, 'mnist.npz')) as f:
        images, labels = f['x_%s' % split], f['y_%s' % split]
        # image: 0 ~ 255 uint8
        # labels 0 ~ 9 uint8
    assert images.shape[0] == labels.shape[0]
    specs['total_size'] = int(images.shape[0])

    """Process np arrary"""
    # set seed
    np.random.seed(seed)
    # sort by labels to get the index permutation
    # classes: 0 1 2 3 4 5 6 7 8 9
    if split == 'train':
        indices = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    elif split == 'test':
        indices = [0, 980, 2115, 3147, 4157, 5139, 6031, 6989, 8017, 8991, 10000]
    perm = labels.argsort()
    images = images[perm]
    labels = labels[perm]
    
    sampled_idc_lists = [] # [list of indices for 0, ... for 1, ...]
    for i in range(specs['num_classes']):
        sampled_idc_lists.append(
            np.random.randint(
                low=indices[i],
                high=indices[i+1],
                size=max_epochs).tolist())
    sampled_idc_mat = np.array(sampled_idc_lists)
    sampled_idc_mat = np.transpose(sampled_idc_mat, [1, 0])
    sampled_idc_lists = sampled_idc_mat.flatten().tolist()
    assert len(sampled_idc_lists) == max_epochs * specs['num_classes']
    # we let n_repeats = steps_per_epoch = number of computed gradients
    list_of_images = []
    list_of_labels = []
    for idx in sampled_idc_lists:
        for _ in range(n_repeats):
            list_of_images.append(images[idx])
            list_of_labels.append(labels[idx])
    res_images = np.stack(list_of_images, axis=0)
    res_labels = np.array(list_of_labels)
    assert res_images.shape == (max_epochs*specs['num_classes']*n_repeats, specs['image_size'], specs['image_size'])
    assert res_labels.shape == (max_epochs*specs['num_classes']*n_repeats,)

    specs['total_size'] = res_labels.shape[0]
    return (res_images, res_labels), specs

def inputs(split, data_dir, max_epochs, n_repeats, cropped_size,
           seed=123, total_batch_size=1):
    """Construct mnist inputs for dream experiment.

    Args:
        split: 'train' or 'test' split to read from dataset;
        data_dir: path to mnist data directory;
        max_epochs: maximum epochs to go through the model;
        n_repeats: number of computed gradients / number of the same input to repeat;
        cropped_size: image size after cropping;
        seed: seed to produce pseudo randomness that we can replicate each time.
        total_batch_size: total number of images per batch.
    Returns:    
        batched_features: a dictionary of the input data features.
    """
    assert split == 'train' or split == 'test'
    
    """Load sampled images and labels"""
    (images, labels), specs = _dream_sample_pairs(
        split, data_dir, max_epochs, n_repeats, seed, total_batch_size)

    if cropped_size == None:
        cropped_size = specs['image_size']
    assert cropped_size <= specs['image_size']

    """Process dataset object"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.prefetch(1)
    dataset = dataset.map(
        lambda image, label:_dream_cropping(image, label, specs, cropped_size), 
        num_parallel_calls=3)
    batched_dataset = dataset.batch(specs['batch_size'])
    batched_dataset = batched_dataset.map(_dream_process, num_parallel_calls=3)
    batched_dataset = batched_dataset.prefetch(1)
    
    specs['image_size'] = cropped_size

    return batched_dataset, specs
