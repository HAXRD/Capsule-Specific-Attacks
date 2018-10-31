# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

import tensorflow as tf 
from models import model
from models.layers import utils
from models.layers import variables
from models.layers import capsule_utils

class CapsuleModel(model.Model):
    """A single GPU model with capsule layers.

    The inference graph includes a 256-channel ReLU convolutional layer,
    a 32x8(rank)-Squash convolutional layer and a 10x8(rank) Squash with routing 
    Capsule layer.
    """

    def _remake(self, capsule_embedding, batched_images, batched_labels): # TODO: undone.
        """Adds the reconstruction subnetwork to build the remakes.

        This subnetwork shares the variables between different target remakes.
        It adds the subnetwork for the first target and reuses the weight 
        variables for the second one.

        Args:
            capsule_embedding: a 3R tensor of shape (batch, 10, 16) containing 
                network embeddings for each digit in the image if present.
            batched_images: placeholder reference for input batched images.
            batched_labels: placeholder reference for input batched labels.
        Returns:
            A list of network remakes of the targets.
        """
        # Image specs
        image_size = self._specs['image_size']
        image_depth = self._specs['depth']

        num_pixels = image_depth * image_size * image_size

        with tf.name_scope('recons'):
            remake = capsule_utils.reconstruction(
                capsule_mask=batched_labels,
                num_atoms=16,
                capsule_embedding=capsule_embedding,
                layer_sizes=[512, 1024],
                num_pixels=num_pixels,
                reuse=False,
                image=batched_images,
                balance_factor=0.0005)
        
        return remake

    def _build_capsule(self, input_tensor, num_classes, tower_idx):
        """Adds capsule layers.

        A slim convolutional capsule layer transforms the input tensor to capsule
        format. The nonlinearity for slim convolutional capsule is squash function
        but there is no routing and each sptial instantiation of capsule is 
        derived from tranditional convolution operations.
        In order to connect the convolutional capsule layer to its next fully
        connected capsule layer, the grid position of convolutional capsule is
        merged with different types of capsules' dimmensions and capsule2 learns
        different transformations for possible `capsule space` arrangement.

        Args:
            input_tensor: 5 rank input tensor, shape (batch, 1, 256, h, w)
            num_classes: number of object categories. Used as the output dimmension.
        Returns:
            A 3R tensor of the next capsule layer with 10 capsule embeddings.
        """
        capsule1 = capsule_utils.conv_slim_capsule(
            tower_idx,
            input_tensor,
            in_dim=1,
            in_atoms=256,
            out_dim=self._hparams.num_prime_capsules,
            out_atoms=8, 
            layer_name='conv_capsule1',
            kernel_size=9,
            stride=2,
            padding=self._hparams.padding,
            num_routing=1,
            leaky=self._hparams.leaky)
        capsule1_atom_last = tf.transpose(capsule1, [0, 1, 3, 4, 2])
        capsule1_3d = tf.reshape(capsule1_atom_last,
                                 [tf.shape(input_tensor)[0], -1, 8])
        _, _, _, height, width = capsule1.get_shape()
        in_dim = self._hparams.num_prime_capsules * height.value * width.value
        
        return capsule_utils.capsule(
            tower_idx,
            in_tensor=capsule1_3d, 
            in_dim=in_dim, 
            in_atoms=8,
            out_dim=num_classes, 
            out_atoms=16, 
            layer_name='capsule2',
            num_routing=self._hparams.routing,
            leaky=self._hparams.leaky)

    def build_replica(self, tower_idx):
        """Adds a replica graph ops.

        Builds the architecture of the neural net to derive logits from 
        batched_dataset. The inference graph defined here should involve 
        trainable variables otherwise the optimizer will raise a ValueError.

        Args:
            tower_idx: the index number for this tower. Each tower is named
                as tower_{tower_idx} and resides on gpu:{tower_idx}.
        Returns:
            Inferred namedtuple containing (logits, recons).
        """
        # Image specs
        image_size = self._specs['image_size']
        image_depth = self._specs['depth']
        num_classes = self._specs['num_classes']

        # Define input_tensor for input batched_images
        batched_images = tf.placeholder(tf.float32,
            shape=[None, image_depth, image_size, image_size],
            name='batched_images') # (?, 3, h, w)
        """visual"""
        tf.add_to_collection('tower_%d_batched_images' % tower_idx, batched_images)

        # ReLU Convolution
        with tf.variable_scope('conv1') as scope:
            kernel = variables.weight_variable(
                shape=[9, 9, image_depth, 256],
                stddev=5e-2,
                verbose=self._hparams.verbose)
            biases = variables.bias_variable(
                [256], verbose=self._hparams.verbose)
            conv1 = tf.nn.conv2d(
                batched_images,
                kernel, strides=[1, 1, 1, 1],
                padding=self._hparams.padding,
                data_format='NCHW')
            pre_activation = tf.nn.bias_add(conv1, biases, 
                data_format='NCHW', name='logits')
            """visual"""
            tf.add_to_collection('tower_%d_visual' % tower_idx, pre_activation)
            relu1 = tf.nn.relu(pre_activation, name=scope.name)
            if self._hparams.verbose:
                tf.summary.histogram(scope.name + '/activation', relu1)
        hidden1 = tf.expand_dims(relu1, 1) # (?, 1, 3, h, w) h,w are different from previous ones.

        # Capsules
        capsule_output = self._build_capsule(hidden1, num_classes, tower_idx)
        logits = tf.norm(capsule_output, axis=-1, name='logits')
        """visual"""
        tf.add_to_collection('tower_%d_visual' % tower_idx, logits)

        # Declare one-hot format placeholder for batched_labels
        batched_labels = tf.placeholder(tf.int32,
            shape=[None, num_classes], name='batched_labels')
        tf.add_to_collection('tower_%d_batched_labels' % tower_idx, batched_labels)

        # Reconstruction
        remake = None
        if self._hparams.remake:
            remake = self._remake(capsule_output, batched_images, batched_labels)
        else:
            remake = None
        
        return model.Inferred(logits, remake)
