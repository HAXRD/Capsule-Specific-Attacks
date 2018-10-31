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

def _process(batched_noise_imgs):
    feature = {
        'images': batched_noise_imgs
    }
    return feature

def inputs(max_epochs, steps_per_epoch=1, depth=3, split='noise', batch_size=1, seed=123):
    """Construct input for layer visualization.
    
    We the noise images are produced as follows:
        1. Initialization a noise image with a fixed seed to 
        produce pseudo randomness.
        2. The total number of the noise images
        = max_epochs * steps_per_epoch * batch_size(=1)
        = max_epochs * steps_per_epoch

    Args:
        max_epochs: number of epochs to create.
        steps_per_epoch: number of batches_per_epoch.
        depth: number of channels of noise images.
        split: 'noise'
        batch_size: should be 1 by all means.
        seed: seed to reproduce pseudo randomness.
    """
    assert steps_per_epoch != None
    """Dataset specs"""
    specs = {
        'split': split,
        'total_size': max_epochs * steps_per_epoch * batch_size,
        'steps_per_epoch': steps_per_epoch,

        'batch_size': batch_size,
        'max_epochs': max_epochs,

        'image_size': 24,
        'depth': depth
    }

    """Set random seed"""
    np.random.seed(seed)

    """Initialize a grey image with noise"""
    noise_imgs = np.random.uniform(size=(
        specs['batch_size'], 
        specs['depth'], specs['image_size'], specs['image_size'])) + 100.0
    """Convert into 0. ~ 1. and normalize them"""
    noise_imgs = noise_imgs * (1. / 255.)
    # noise_imgs = (noise_imgs - noise_imgs.mean()) / noise_imgs.std()

    """Process dataset object"""
    # extract single instance
    t_noise_img = tf.data.Dataset.from_tensor_slices((noise_imgs)) # total_size=1
    # create `max_epochs*steps_per_epoch` number of copys
    t_noise_imgs = t_noise_img.repeat(
        specs['max_epochs']*specs['steps_per_epoch'])
    # create batched image tensors
    batched_noise_imgs = t_noise_imgs.batch(specs['batch_size'])
    # convert to feature
    batched_dataset = batched_noise_imgs.map(_process)
    # prefetch 1
    batched_dataset = batched_dataset.prefetch(1)

    return batched_dataset, specs

if __name__ == '__main__':
    dataset, _ = inputs(1, 1, 1)
    iterator = dataset.make_initializable_iterator()
    next_features = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        single = sess.run(next_features)
        print(single['images'])

        import matplotlib.pyplot as plt 
        img = np.squeeze(single['images'])
        plt.imshow(img, cmap='gray')
        plt.show()
