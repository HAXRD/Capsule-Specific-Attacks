# Copyright 2018 Xu Chen All Rights Reserved.
# Credit https://github.com/zalandoresearch/fashion-mnist#get-the-data
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
import random
import gzip

def _single_process(image, label, specs):
    """Map function to process single instance of dataset object.
    
    Args: 
        image: numpy array image object, (28, 28), 0 ~ 255 uint8
        label: numpy array label, (,)
        specs: dataset specifications
    Returns:
        feature: a dictionary contains image, label, recons_image, recons_label.
    """
    # expand image dimensions into (HWC)
    image = tf.expand_dims(image, -1)
    if specs['distort']:
        resized_size = 24
        # resize images
        image = tf.image.resize_images(image, [resized_size, resized_size])
        if specs['split'] == 'train':
            # random rotation within -15° ~ 15°
            image = tf.contrib.image.rotate(
                image, random.uniform(-0.26179938779, 0.26179938779))
    
    # convert from 0 ~ 255 to 0. ~ 1.
    image = tf.cast(image, tf.float32) * (1. / 255.)
    # transpose image into (CHW)
    image = tf.transpose(image, [2, 0, 1]) # (CHW)
    
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

def load_fashion_mnist(path, split='train'):
    """Load fashion-mnist dataset from byte files.

    Args:
        path: given directory of where the dataset was stored
        split: 'train' or 'test'.
    Return:
        images: numpy array storing image data, uint8 0 ~ 255, (60000 or 10000, 28, 28)
        labels: numpy array storing label data, uint8 0 ~ 9, (60000 or 10000,)
    """
    if split == 'test':
        split = 't10k'
    labels_path = os.path.join(
        path, '%s-labels-idx1-ubyte.gz' % split)
    images_path = os.path.join(
        path, '%s-images-idx3-ubyte.gz' % split)

    # read label data    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(
            lbpath.read(), dtype=np.uint8, offset=8)
    # read image data
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels 


def inputs(total_batch_size, num_gpus, max_epochs,
           data_dir, split, distort=True):
    """Construct inputs for fashion mnist dataset.

    Args:
        total_batch_size: total number of images per batch.
        num_gpus: number of GPUs available to use.
        max_epochs: maximum epochs to go through the model.
        data_dir: path to the fashion-mnist data directory.
        split: 'train' or 'test', which split of dataset to read from.
        distort: whether to distort the iamges, including scale down the image and rotations.
    Returns:
        batched_dataset: Dataset object, each instance is a feature dictionary
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

        'image_size': 28,
        'depth': 1,
        'num_classes': 10,
        'distort': distort
    }

    """Load data from downloaded byte file"""
    images, labels = load_fashion_mnist(data_dir, split)
    # image: 0 ~ 255 uint8
    # label: 0 ~ 9 uint8
    assert images.shape[0] == labels.shape[0]
    specs['total_size'] = int(images.shape[0])
    specs['steps_per_epoch'] = int(specs['total_size'] // specs['total_batch_size'])

    """Process dataset object"""
    # read from numpy array
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)) # ((28, 28), (,))
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
        lambda image, label: _single_process(image, label, specs),
        num_parallel_calls=3)
    specs['image_size'] = 24 # after processed single example, the image size
                             # will be resized to 24
    # stack into batches
    batched_dataset = dataset.batch(specs['batch_size'])
    # process into feature
    batched_dataset = batched_dataset.map(
        _feature_process,
        num_parallel_calls=3)
    # prefetch to improve the performance
    batched_dataset = batched_dataset.prefetch(specs['num_gpus'])

    return batched_dataset, specs

if __name__ == '__main__':
    ######### debug for train and test #########
    total_batch_size, num_gpus, max_epochs = 2, 2, 1
    data_dir, split, distort = '/Users/xu/Downloads/fashion-mnist', 'test', True

    dataset, specs = inputs(total_batch_size, num_gpus, max_epochs,
                            data_dir, split, distort)
    iterator = dataset.make_initializable_iterator()
    next_feature = iterator.get_next()
    
    from pprint import pprint
    pprint(specs)

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        single = sess.run(next_feature)
        print(single['images'])
        print(single['labels'].shape)

        import matplotlib.pyplot as plt 
        img = np.squeeze(single['images'])
        plt.imshow(img, cmap='gray')
        plt.show()
    ###########
    
    # images, labels = load_fashion_mnist('/Users/xu/Downloads/fashion-mnist', 'test')
    # print(images, labels)
    # print(images.shape, labels.shape)
    
    # import matplotlib.pyplot as plt 
    # img = np.reshape(images[0], (28, 28))
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # pass