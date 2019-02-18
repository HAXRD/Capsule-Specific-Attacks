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

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_gpus', 2,
                        'Number of GPUs available.')
tf.flags.DEFINE_integer('total_batch_size', 1,
                        'The total batch size for each batch. It will be splitted into num_gpus partitions.')
tf.flags.DEFINE_integer('save_epochs', 10,
                        'How often to save ckpt files.')
tf.flags.DEFINE_integer('max_epochs', 10,
                        'train, evaluate, ensemble: maximum epochs to run;\n'
                        'others: number of different examples to viasualize.')
tf.flags.DEFINE_integer('image_size', 28,
                        'Define the image size for dataset.')

tf.flags.DEFINE_string('mode', 'train',
                       'train: train the model;\n'
                       'evaluate: evaluate the model for both training and testing set using different evaluation metrics;\n'
                       'glitch: find examples the were predicted into wrong class;\n'
                       'Capsule Norm:\n'
                       '    naive_max_norm, max_norm_diff,\n'
                       '    noise_naive_max_norm,noise_max_norm_diff;\n'
                       'Capsule Direction:\n'
                       '    naive_max_caps_dim, max_caps_dim_diff,\n'
                       '    noise_naive_max_caps_dim, noise_max_caps_dim_diff;\n')
tf.flags.DEFINE_string('hparams_override', None,
                       '--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false')
tf.flags.DEFINE_string('data_dir', None,
                       'The data directory')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.\n'
                       'mnist, fashion_mnist, svhn, cifar10.')
tf.flags.DEFINE_string('model', 'cap',
                       'The model to use for the experiment.\n'
                       'cap or cnn.')
tf.flags.DEFINE_string('summary_dir', './summary',
                       'The directory to write results.')

###### Norm & Direction only ######
tf.flags.DEFINE_integer('iter_n', 100,
                        'Number of iteration to run the gradient ascent;\n'
                        'the code only record iterations at\n'
                        '[1, 2, 3, 4, 5, 6, 7, 8, 9,\n'
                        ' 10, 20, 40, 60, 80,\n'
                        ' 100, 200, 400, 600, 800, 1000].')
tf.flags.DEFINE_string('step', '0.004',
                       'Step size for each iteration.')
###################################
tf.flags.DEFINE_string('threshold', '0.0',
                       'Capsule Norm, Capsule Direction:\n'
                       '    Only those standardized gradients that larger than the threshold will be added during gradient ascent;\n'
                       'ensemble: \n'
                       '    Those contribution coefficient c_ji of part capsules j to the target whole capsule i that larger than the threshold will be set to zeros in order to remove the effect of class type j from the rest of capsules k, where k≠j.')

def default_hparams():
    """Builds an HParams object with default hperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=2000,
        leaky=False,
        learning_rate=0.001,
        loss_type='margin',
        num_prime_capsules=32,
        padding='VALID',
        remake=True,
        routing=3,
        verbose=False)