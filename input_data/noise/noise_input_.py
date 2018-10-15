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

def inputs(num_gpus, n_repeats, total_size=50, seed=123,split='default'):
    """Construct input for layer visualization.

    Here we let `total_size` = `total_batch_size` so that each epoch
    only contains single batch when `num_gpus`=1 (minibatch could
    be created if `num_gpus` > 1, but the idea stays the same).
    1 batch/epoch contains `total_size` number of different noise
        --> compare across 1 batch/epoch, we can find the effect of 
        different noise initializations on layer visualization.
    Repeat the batch/epoch `n_repeats` times
        --> we use `n_repeats` batch/epoch initializations to compare
        the different between each layer we want to visualization.
    
    Args:
        num_gpus: number of GPUs available.
        n_repeats: number of epochs to create.
        total_size: number of different initializations for a epoch.
        seed: seed to reproduce pseudo randomness.
        split: 'default'
    """
    # Dataset specs
    specs = {
        'split': split,
        'n_repeats': n_repeats,
        'total_batch_size': total_size,
        'num_gpus': num_gpus,
        'batch_size': total_size // max(1, num_gpus),
        'image_dim': 32,
        'depth': 3,
        'num_classes': 10
    }

    np.random.seed(seed)
    # Initialize a grey image with noise.
    noise_imgs = np.random.uniform(size=(
        specs['batch_size'], specs['image_dim'], 
        specs['image_dim'], specs['depth'])) + 100.0

    # Extract single instance
    t_noise_img = tf.data.Dataset.from_tensor_slices((noise_imgs))
    # Prefetch `total_batch_size` number of instances
    t_noise_imgs = t_noise_img.prefetch(buffer_size=specs['total_batch_size'])
    # Create `n_repeats` number of epochs
    t_noise_imgs = t_noise_imgs.repeat(specs['n_repeats'])
    # Create batched image tensors
    batched_noise_imgs = t_noise_imgs.batch(specs['batch_size'])

    return batched_noise_imgs, specs

if __name__ == '__main__':
    dataset, _ = inputs(2, 1)
    iterator = dataset.make_initializable_iterator()
    next_features = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        try:
            single = sess.run(next_features)
            print(single.shape)
        except tf.errors.OutOfRangeError:
            pass