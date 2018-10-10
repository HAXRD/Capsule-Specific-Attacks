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

import abc 
import collections
import tensorflow as tf 
# from models.layers import layers

Inferred = collections.namedtuple('Inferred',
                                  ('logits', 'remakes'))

class Model(object):
    """Base class for building a model and running inference on it."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams):
        """Initializes the model parameters.
        
        Args:
            hparams: The hyperparameters for the model as tf.contrib.train.HParams.
        """
        self._hparams = hparams
        with tf.device('/cpu:0'):
            self._global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)
            
            learning_rate = tf.train.exponential_decay(
                learning_rate=hparams.learning_rate,
                global_step=self._global_step,
                decay_steps=hparams.decay_steps,
                decay_rate=hparams.decay_rate)
            learning_rate = tf.maximum(learning_rate, 1e-6)

            self._optimizer = tf.train.AdamOptimizer(learning_rate)
    
    def build_replica(self):
        """Adds a replica graph ops.

        Builds the architecture of the neural net to derive logits from batched_dataset.
        The inference graph defined here should involve trainable variables
        otherwise the optimizer will raise a ValueError.

        Returns:
            undefined
        """
        raise NotImplementedError('Not implemented.')

    def _build_single_tower(self, tower_idx):
        """Build sinlge replica of the model (we term it as `tower`).

        Args:
            tower_idx: The index of the tower, used for naming.
        """
        with tf.device('/gpu:%d' % tower_idx):
            with tf.name_scope('tower_%d' % (tower_idx)) as scope:
                logits = self.build_replica()# !!! Here
                # TODO: 
                # 1. get all the tensors by their names. write experiment
                # 2. compute those 'logits' gradient
                # 3. evaluate the model.
                # 4. train the model.
        pass

    def build_model_on_multi_gpus(self, num_gpus):
        """Build the model and Graph and add the train ops on multiple GPUs.

        Divide the inference and gradient computation on multiple GPUs, where
        each GPU has its own replica of the model graph. Then user can get 
        whichever tensor by using `tf.get_tensor_by_name` method.

        Args:
            num_gpus: Number of GPUs available.
        """
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                self._build_single_tower(i)
