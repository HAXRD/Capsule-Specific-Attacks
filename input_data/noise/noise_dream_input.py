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
import numpy as np 
import tensorflow as tf 

def _dream_process(image):

    batched_features = {
        'images': image
    }
    return batched_features

def inputs(split, depth, max_epochs, n_repeats,
           seed=123, total_batch_size=1):
    """Construct noise inputs for dream experiment.

    Args:
        split: 'noise' split to read from dataset.
        depth: number of channels.
        max_epochs: maximum epochs to go through the model.
        n_repeats: number of computed gradients / number of the same input to repeat.
        seed: seed to produce pseudo randomness that we can replicate each time.
        total_batch_size: total number of images per batch.
    Returns:    
        batched_features: a dictionary of the input data features.
    """

    """Dataset specs"""
    specs = {
        'split': split,
        'max_epochs': max_epochs,
        'steps_per_epoch': n_repeats,
        'batch_size': total_batch_size,
        'image_size': 24,
        'depth': depth,
        'num_classes': 10
    }

    """Set random seed"""
    np.random.seed(seed)

    """Initialize noise images"""
    noise_img_list = []
    for _ in range(max_epochs):
        one_noise_img = np.random.uniform(size=(
            specs['batch_size'],
            specs['depth'], specs['image_size'], specs['image_size'])) + 127.0
        """Convert into 0. ~ 1. """
        one_noise_img = one_noise_img * (1. / 255.)
        for _ in range(10*n_repeats):
            noise_img_list.append(one_noise_img)
    """Transform into np array"""
    noise_img_matr = np.stack(noise_img_list, axis=0)

    """Process dataset object"""
    # extract single instance 
    dataset = tf.data.Dataset.from_tensor_slices((noise_img_matr))
    # create batched image dataset
    batched_dataset = dataset.batch(specs['batch_size'])
    # convert into feature
    batched_dataset = batched_dataset.map(_dream_process)
    # prefetch 1
    batched_dataset = batched_dataset.prefetch(1)

    return batched_dataset, specs
