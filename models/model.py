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
from models.layers import utils
# from models.layers import layers

Inferred = collections.namedtuple('Inferred',
                                 ('logits', 'remakes'))
TowerResult = collections.namedtuple('TowerResult',
                                    ('inferred', 'train_op', 
                                    'summary', 'correct', 'accuracy'))

class Model(object):
    """Base class for building a model and running inference on it."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams, dataset_specs):
        """Initializes the model parameters.
        
        Args:
            hparams: The hyperparameters for the model as 
                tf.contrib.train.HParams.
            dataset_spec: Specifications of the dataset, including
                `split`, `max_epochs`, `total_batch_size`, `num_gpus`,
                `num_gpus`, `image_dim`, `depth`, 
                `num_classes`, `total_size`, `steps_per_epoch`.
        """
        self._hparams = hparams
        self._specs = dataset_specs
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

        Builds the architecture of the neural net to derive logits from 
        batched_dataset. The inference graph defined here should involve 
        trainable variables otherwise the optimizer will raise a ValueError.

        Returns:
            Inferred namedtuple containing (logits, None).
        """
        raise NotImplementedError('Not implemented.')

    def _build_single_tower(self):
        """Build sinlge replica of the model (we term it as `tower`).

        Returns:
            A
        """
        with tf.device('/gpu:0'):
            with tf.name_scope('tower') as scope:
                # Build a tower (replica)
                inferred = self.build_replica()
                # Calculate the loss and predictions
                total_loss, num_correct_per_batch, accuracy = utils.evaluate(
                    logits=inferred.logits,
                    scope=scope,
                    loss_type=self._hparams.loss_type)
                
                train_op = self._optimizer.minimize(total_loss)
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                summary = tf.summary.merge(summaries)

        return TowerResult(inferred, train_op, 
                           summary, num_correct_per_batch, accuracy)

    def build_model_on_single_gpu(self):
        """Build the model and Graph and add the train ops on single GPUs.

        This single GPU will be allocated with single replica of the model.
        """
        with tf.variable_scope(tf.get_variable_scope()):
            tower_output = self._build_single_tower()
        
        tf.add_to_collection('summary', tower_output.summary)
        tf.add_to_collection('train_op', tower_output.train_op)
        tf.add_to_collection('correct', tower_output.correct)
        tf.add_to_collection('accuracy', tower_output.accuracy)

        return tower_output
        