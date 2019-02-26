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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import tensorflow as tf 

from input_data.cifar10 import load_cifar10_data

def _single_process(image, label, specs, cropped_size):
    """A function to parse each record into an image and an label.

    Args:
        image: numpy array image object (28, 28, 3), 0 ~ 255 uint8;
        label: numpy array label, (,);
        specs: dataset specifications;
        cropped_size: image size after cropping.
    Returns:
        feature: a dictionary contains an image instance and a label instance.    
    """
    # convert from 0 ~ 255 to 0. ~ 1.
    image = tf.cast(image, tf.float32) * (1. / 255.)
    
    if specs['distort']:
        if cropped_size <= specs['image_size']:
            if specs['split'] == 'train':
                # random cropping 
                image = tf.random_crop(image, [cropped_size, cropped_size, 3])
                # random flipping
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=63)
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            elif specs['split'] == 'test':
                # central cropping
                image = tf.image.resize_image_with_crop_or_pad(image, cropped_size, cropped_size)
    
    # transpose image into (CHW)
    image = tf.transpose(image, [2, 0, 1])

    feature = {
        'image': image,
        'label': tf.one_hot(label, 10)
    }
    return feature

def _feature_process(feature):
    """Map function to process batched data inside feature dictionary.

    Args:
        feature: a dictionary contains image, label, recons_image, recons_label.
    Returns:
        batched_feature: a dictionary contains images, labels, recons_images, recons_labels.
    """
    batched_feature = {
        'images': feature['image'],
        'labels': feature['label'],
    }
    return batched_feature

def inputs(total_batch_size, num_gpus, max_epochs, cropped_size,
           data_dir, split, distort=True):
    """Construct inputs for cifar10 dataset.

    Args:
        total_batch_size: total number of images per batch;
        num_gpus: number of GPUs available to use;
        max_epochs: maximum epochs to go through the model;
        cropped_size: image size after cropping;
        data_dir: path to the fashion-mnist data directory;
        split: 'train' or 'test', which split of dataset to read from;
        distort: whether to distort the iamges, including scale down the image and rotations.
    Returns:
        batched_dataset: Dataset object, each instance is a feature dictionary;
        specs: dataset specifications.
    """
    assert split == 'train' or split == 'test'

    """Dataset specs"""
    specs = {
        'split': split,
        'total_size': None, # total size of one epoch
        'steps_per_epoch': None, # number of steps per epoch

        'total_batch_size': int(total_batch_size),
        'num_gpus': num_gpus,
        'batch_size': int(total_batch_size / num_gpus),
        'max_epochs': int(max_epochs), # number of epochs to repeat

        'image_size': 32,
        'depth': 3,
        'num_classes': 10,
        'distort': distort
    }
    
    if cropped_size == None:
        cropped_size = specs['image_size']
    assert cropped_size <= specs['image_size']

    """Load data from mat file"""
    images, labels = load_cifar10_data.load_cifar10(data_dir, split)
    # images: 0 ~ 255 uint8 (?, 32, 32, 3)
    # labels: 0 ~ 9 uint8   (?, 1)
    assert images.shape[0] == labels.shape[0]
    specs['total_size'] = int(images.shape[0])
    specs['steps_per_epoch'] = int(specs['total_size']) // specs['total_batch_size']

    """Process dataset object"""
    # read from numpy array
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # prefetch examples
    dataset = dataset.prefetch(
        buffer_size=specs['batch_size']*specs['num_gpus']*2)
    # shuffle (if 'train') and repeat 'max_epochs'
    if split == 'train':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=specs['batch_size']*specs['num_gpus']*10,
            count=specs['max_epochs']))
    else:
        dataset = dataset.repeat(specs['max_epochs'])
    # process single example
    dataset = dataset.map(
        lambda image, label: _single_process(image, label, specs, cropped_size),
        num_parallel_calls=3)
    specs['image_size'] = cropped_size
    # stack into batches
    batched_dataset = dataset.batch(specs['batch_size'])
    # process into feature
    batched_dataset = batched_dataset.map(
        _feature_process,
        num_parallel_calls=3)
    # prefetch to improve the performance
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs
