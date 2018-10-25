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

def _cropping(image, label):
    
    cropped_image_dim = 24
    image = tf.expand_dims(image, -1) # (HWC)
    image = tf.image.resize_image_with_crop_or_pad(image, cropped_image_dim, cropped_image_dim)
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, [2, 0, 1]) # (CHW)

    feature = {
        'image': image, 
        'label': tf.one_hot(label, 10),
        'recons_image': image,
        'recons_label': label
    }
    return feature

def _processing(feature):
    
    batched_features = {
        'images': feature['image'],
        'recons_images': feature['recons_image'],
        'labels': feature['label'],
        'recons_labels': feature['recons_label']
    }
    return batched_features

def inputs(split, data_dir, batch_size, max_epochs):
    """Construct input for mnist experiment.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the mnist tfrecords data directory.
        batch_size: total number of images per batch.
        max_epochs: maximum epochs to go through the model.
    Returns:
        batched_features: a dictionary of the input data features.
    """
    # Dataset specs
    specs = {
        'split': split,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'image_dim': 28, 
        'depth': 1, 
        'num_classes': 10
    }
    
    with np.load(os.path.join(data_dir, 'mnist.npz')) as f:
        if split == 'train':
            images, labels = f['x_train'], f['y_train']
            specs['total_size'] = 60000
        else:
            images, labels = f['x_test'], f['y_test']
            specs['total_size'] = 10000
    specs['steps_per_epoch'] = specs['total_size'] // specs['batch_size'] 
    assert images.shape[0] == labels.shape[0]
    images = images / 255.0
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)) # ((28, 28), (,))
    dataset = dataset.prefetch(buffer_size=specs['batch_size']*10)

    if split == 'train':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=specs['batch_size']*10, count=specs['max_epochs']))
    elif split == 'test':
        dataset = dataset.repeat(specs['max_epochs'])
    # crop the images the dataset
    dataset = dataset.map(_cropping, num_parallel_calls=3)
    # stack into batches
    batched_dataset = dataset.batch(specs['batch_size'])
    # Process into feature
    batched_dataset = batched_dataset.map(_processing, num_parallel_calls=3)
    # Prefetch to improve the performance
    batched_dataset = batched_dataset.prefetch(1)
    specs['image_dim'] = 24
    return batched_dataset, specs

def _dream_cropping(image, label):
    
    cropped_image_dim = 24
    image = tf.expand_dims(image, -1) # (HWC)
    image = tf.image.resize_image_with_crop_or_pad(image, cropped_image_dim, cropped_image_dim)
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, [2, 0, 1]) # (CHW)

    feature = {
        'image': image, 
        'label': label
    }
    return feature

def _dream_processing(feature):

    batched_features = {
        'images': feature['image'],
        'labels': feature['label'],
    }
    return batched_features

def _dream_sample_pairs(split, data_dir, max_epochs, steps_per_epoch, 
                        seed=123, batch_size=1):
    """
    We do the following steps to produce the dataset:
        1. sample one (image, label) pair in one class;
        2. repeat pair in 1. {steps_per_epoch} times;
        3. go back to do 1. unless we finish one iteration 
           (after a {num_classes} time loop). And we consider
           this as one epoch.
        4. go back to do 1. again to finish {max_epochs} loop.
    So there will be {max_epochs} number of unique pairs selected for 
    each class.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the mnist tfrecords data directory.
        max_epochs: maximum epochs to go through the model.
        steps_per_epoch: number of computed gradients
        seed: seed to produce pseudo randomness.
        batch_size: total number of images per batch.
    Returns:
        processed images, labels and specs
    """
    assert max_epochs < 100
    # Dataset specs
    specs = {
        'split': split, 
        'max_epochs': max_epochs,
        'steps_per_epoch': steps_per_epoch,
        'batch_size': batch_size,
        'image_dim': 28,
        'depth': 1,
        'num_classes': 10
    }

    # Set seed
    np.random.seed(seed)

    # Load mnist train/test set
    with np.load(os.path.join(data_dir, 'mnist.npz')) as f:
        images, labels = f['x_train'], f['y_train']
        # images, labels = f['x_test'], f['y_test']
        assert images.shape[0] == labels.shape[0]

    # sort by labels to get the index permutation
    # class: 0    1    2    3    4    5    6    7    8    9
    indices = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000] # train
    # indices = [0, 980, 2115, 3147, 4157, 5139, 6031, 6989, 8017, 8991, 10000] # test
    perm = labels.argsort() 
    images = images[perm]
    labels = labels[perm]
    assert images.shape == (60000, 28, 28)
    assert labels.shape == (60000,)

    sampled_idc_lists = [] # [list of idc for 0, ... for 1, ...]
    for i in range(specs['num_classes']):
        sampled_idc_lists.append(
            np.random.randint(
                low=indices[i], 
                high=indices[i+1], 
                size=max_epochs).tolist())
    sampled_idc_mat = np.array(sampled_idc_lists)
    assert sampled_idc_mat.shape == (specs['num_classes'], max_epochs)
    sampled_idc_mat = np.transpose(sampled_idc_mat, [1, 0])
    assert sampled_idc_mat.shape == (max_epochs, specs['num_classes'])
    sampled_idc_lists = sampled_idc_mat.flatten().tolist() # [lists of class idc for each epoch] flattened
    assert len(sampled_idc_lists) == max_epochs*specs['num_classes']

    # !!! we let steps_per_epoch=number of computed gradients
    list_of_images = []
    list_of_labels = []
    for idx in sampled_idc_lists:
        for _ in range(steps_per_epoch):
            list_of_images.append(images[idx])
            list_of_labels.append(labels[idx])
    res_images = np.stack(list_of_images, axis=0)
    res_labels = np.array(list_of_labels)
    assert res_images.shape == (max_epochs*specs['num_classes']*steps_per_epoch, specs['image_dim'], specs['image_dim'])
    assert res_labels.shape == (max_epochs*specs['num_classes']*steps_per_epoch,)

    specs['total_size'] = res_labels.shape[0]
    return (res_images, res_labels), specs

def dream_inputs(split, data_dir, max_epochs, steps_per_epoch, 
                 seed=123, batch_size=1):
    """Construct input for mnist dream experiment.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the mnist tfrecords data directory.
        max_epochs: maximum epochs to go through the model.
        steps_per_epoch: number of computed gradients
        seed: seed to produce pseudo randomness.
        batch_size: total number of images per batch.
    Returns:
        batched_features: a dictionary of the input data features.
    """
    # Get sampled images and labels
    (images, labels), specs = _dream_sample_pairs(
        split, data_dir, max_epochs, steps_per_epoch, seed, batch_size)

    images = images / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)) 
    dataset = dataset.prefetch(buffer_size=specs['batch_size']*specs['num_classes'])
    dataset = dataset.map(_dream_cropping, num_parallel_calls=3)
    batched_dataset = dataset.batch(specs['batch_size'])
    batched_dataset = batched_dataset.map(_dream_processing, num_parallel_calls=3)
    batched_dataset = batched_dataset.prefetch(1)

    specs['image_dim'] = 24
    return batched_dataset, specs

if __name__ == '__main__':
    #############################################
    """
    max_epochs = 2
    steps_per_epoch = 2
    dataset, specs = dream_inputs('dream', '/Users/xu/Downloads/mnist',
                                  max_epochs, steps_per_epoch)
    iterator = dataset.make_initializable_iterator()
    next_feature = iterator.get_next()
    
    print(specs)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10*max_epochs*steps_per_epoch):
            single = sess.run(next_feature)

            if i + 1 == 10 * steps_per_epoch + steps_per_epoch * 7 + 2:
                print('i: ', i)
                print(single['images'].shape)
                print(single['labels'])
                # print(single['recons_images'].shape)
                # print(single['recons_labels'])

                import matplotlib.pyplot as plt
                import numpy as np
                img = np.squeeze(single['images'])
                plt.imshow(img, cmap='gray')
                plt.show()
    """    
    #############################################
    
    pass