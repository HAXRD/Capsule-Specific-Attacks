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

def _parse_record(record):
    """A function to parse each record into image and label.

    Args:
        record: a single record instance.
    Returns:
        image: a single image
        label: a corresponding label
    """
    # record specs
    image_dim = 32
    depth = 3
    image_bytes = image_dim * image_dim * depth
    label_bytes = 1
    record_bytes = label_bytes + image_bytes

    # decode binary into uint8
    uint_data = tf.decode_raw(record, tf.uint8)

    # get label
    label = tf.cast(tf.strided_slice(uint_data, [0], [label_bytes]), tf.int32)
    label.set_shape([1])
    # get image
    depth_major_image = tf.reshape(
        tf.strided_slice(uint_data, [label_bytes], [record_bytes]),
        [depth, image_dim, image_dim])
    image = tf.cast(tf.transpose(depth_major_image, [1, 2, 0]), tf.float32)
    # !!! taking out distort, cropping, but still keeping standardization
    image = tf.image.per_image_standardization(image) # this func requires (HWC)
    # revert back to (CHW)
    image = tf.transpose(image, [2, 0, 1])

    feature = {
        'image': image,
        'label': tf.one_hot(label, 10),
        'recons_image': image,
        'recons_label': label
    }

    return feature

def _process_batched_features(feature):
    """A function process the features.

    Args:
        feature: A dictionary containing 'image', 'label', 'recons_image',
            'recons_label'. (actually now all the variables should plural,
            since we stacked them into batches).
    Returns:
        batched_features: A dictionary containing a batch_size of 'images',
            'labels', 'recons_images', 'recons_labels'.
    """
    batched_features = {
        'images': feature['image'],
        'recons_images': feature['recons_image'],
        'labels': tf.squeeze(feature['label'], [1]),
        'recons_labels': tf.squeeze(feature['recons_label'], [1])
    }
    return batched_features

def inputs(split, data_dir, total_batch_size, num_gpus, max_epochs):
    """Construct input for cifar10 experiment.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the cifar10 data directory.
        total_batch_size: total number of images per batch.
        num_gpus: number of GPUs available.
        max_epochs: maximum epochs go through the model.
    Returns:
        batched_features: a dictionary of the input data features.
    """
    # Dataset specs
    specs = {
        'split': split,
        'max_epochs': max_epochs,
        'total_batch_size': total_batch_size,
        'num_gpus': num_gpus,
        'batch_size': total_batch_size // max(1, num_gpus),
        'image_dim': 32,
        'depth': 3,
        'num_classes': 10
    }

    # Aggregating the filenames
    if split == 'train':
        filenames = [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, 6)]
        specs['total_size'] = 50000
    elif split == 'test':
        filenames = [
            os.path.join(data_dir, 'test_batch.bin')]
        specs['total_size'] = 10000
    specs['steps_per_epoch'] = specs['total_size'] // specs['total_batch_size']

    # Fixed Length Record Dataset specifications
    image_bytes = specs['image_dim'] * specs['image_dim'] * specs['depth']
    label_bytes = 1
    record_bytes = label_bytes + image_bytes
    
    # Declare dataset object
    dataset = tf.data.FixedLengthRecordDataset(
        filenames, record_bytes)
    # Prefetch the dataset, performance purpose
    dataset = dataset.prefetch(buffer_size=specs['total_batch_size']*10)
    
    # Shuffle(train only) and Repeat the dataset size to 'max_epochs'*total_size    
    if split == 'train':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=specs['total_batch_size']*10, count=specs['max_epochs']))
    elif split == 'test':
        dataset = dataset.repeat(specs['max_epochs'])
    # Parse dataset
    dataset = dataset.map(_parse_record, num_parallel_calls=3)
    # Stack into batches
    batched_dataset = dataset.batch(specs['batch_size'])
    # Process batched_dataset
    batched_dataset = batched_dataset.map(_process_batched_features, num_parallel_calls=3)
    # Prefetch to improve performance
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs

if __name__ == '__main__':
    dataset, _ = inputs('test', '/Users/xu/Downloads/cifar-10-batches-bin', 1, 1, 2)
    iterator = dataset.make_initializable_iterator()
    next_features = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        try:
            single = sess.run(next_features)
            print(single['images'])
            print(single['labels'])
            print(single['recons_labels'])
            import matplotlib.pyplot as plt
            import numpy as np
            img = np.transpose(np.squeeze(single['images']), [1,2,0])
            plt.imshow(img)
            plt.show()

        except tf.errors.OutOfRangeError:
            pass

