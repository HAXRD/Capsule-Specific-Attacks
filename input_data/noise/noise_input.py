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

"""Input utility functions for reading originally noisy data.

Handles reading from noise produced by numpy package. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
from io import BytesIO
import PIL.Image

import tensorflow as tf 

def _batch_features(image, batch_size, image_size):

    image = tf.transpose(image, [2, 0, 1])
    features = {
        'images': image
    }
    batched_features = tf.train.batch(
        features,
        batch_size=batch_size,
        num_threads=1,
        capacity=100+3*batch_size)
    batched_features['height'] = image_size
    batched_features['depth'] = 3
    batched_features['num_targets'] = 1
    batched_features['num_classes'] = 10
    return batched_features

def visstd(arr, step=0.1):
    """Normalize the image range for visualization."""
    return (arr - arr.mean())/max(arr.std(), 1e-4) * step + 0.5

def show_array(arr, fmt='jpeg', name='', show=False):
    arr = np.uint8(np.clip(arr, 0, 1) * 255)
    f = BytesIO()
    img = PIL.Image.fromarray(arr)
    if len(name) > 0:
        img.save(name + '.' + fmt, format=fmt)
    if show:
        img.show()
    
def inputs(size=1, image_size=24, depth=3):
    """Constructs input for noise input. 

    All the noisy images are produced by numpy packages and 
    they are identical and determined by seed.

    Args:
        size: number of noise images to produce.
        image_size: height/width of the image inputs.
        seed: the seed for numpy random function.
    Returns:
        a queue containing size={size} of features
    """
    image_noise = np.random.uniform(size=(image_size, image_size, depth))
    image = tf.image.per_image_standardization(image_noise)
    
    batch_size = 1
    return _batch_features(image, batch_size, image_size)

if __name__ == '__main__':
    inputs(1, 512, 3)