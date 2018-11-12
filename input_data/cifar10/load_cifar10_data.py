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
import os
import numpy as np 
from scipy.io import loadmat

def get_info(data_dir, split='train'):
    """Get cifar10 filenames.

    Args:
        data_dir: data directory of where cifar10 was stored.
        split: 'train' or 'test' split.
    Returns:
        filenames: a list of filenames of tfrecords.
    """

    """Aggregating the filenames"""
    if split == 'train':
        filenames = [
            os.path.join(data_dir, 'data_batch_%d.mat') % i
            for i in range(1, 6)]
    elif split == 'test':
        filenames = [
            os.path.join(data_dir, 'test_batch.mat')]

    return filenames

def load_cifar10(data_dir, split='train'):
    """Get cifar10 data.

    Args:
        data_dir: data directory of where cifar10 was stored.
        split: 'train' or 'test' split.
    Returns:
        images, labels
    """
    filenames = get_info(data_dir, split)
    
    images_list = []
    labels_list = []
    for fn in filenames:
        mat = loadmat(fn)
        images = np.transpose(np.reshape(mat['data'], [-1, 3, 32, 32]), [0, 2, 3, 1])
        labels = np.reshape(mat['labels'], -1)
        images_list.append(images)
        labels_list.append(labels)
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return images, labels
