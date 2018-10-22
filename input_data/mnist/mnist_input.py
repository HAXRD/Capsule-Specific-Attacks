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
mnist = tf.keras.datasets.mnist

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

def inputs(split, batch_size, max_epochs):
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
        'depth': 3, 
        'num_classes': 10
    }

    if split == 'train':
        (images, labels), (_, _) = mnist.load_data()
        specs['total_size'] = 60000
    else:
        (_, _), (images, labels) = mnist.load_data()
        specs['total_size'] = 10000
    
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

if __name__ == '__main__':
    
    dataset, specs = inputs('test', 10, 2)
    iterator = dataset.make_initializable_iterator()
    next_feature = iterator.get_next()
    
    print(specs)

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        single = sess.run(next_feature)
        print(single['images'].shape)
        print(single['labels'])
        print(single['recons_images'].shape)
        print(single['recons_labels'])

        import matplotlib.pyplot as plt
        import numpy as np
        img = np.squeeze(single['images'][3])
        plt.imshow(img, cmap='gray')
        plt.show()
    pass