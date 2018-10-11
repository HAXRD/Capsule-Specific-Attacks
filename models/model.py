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
                                    ('inferred', 'correct', 'total_loss_grad'))
JoinedResult = collections.namedtuple('JoinedResult',
                                     ('summary', 'train_op', 'correct', 'accuracy'))

class Model(object):
    """Base class for building a model and running inference on it."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams, dataset_specs):
        """Initializes the model parameters.
        
        Args:
            hparams: The hyperparameters for the model as 
                tf.contrib.train.HParams.
            dataset_spec: Specifications of the dataset, including
                `split`, `total_batch_size`, `image_dim`, `depth`, 
                `num_classes`, `total_size`.
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
    
    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each variable across all towers.

        Args:
            tower_grads: list, a list of gradient lists for each tower. Each 
                gradient list is a list of (gradient, variable) tuples for 
                all variables.
        Returns:
            average_grads: list, a list of pairs of (gradient, variable) where 
                the gradient has been averaged across all towers.
        """
        average_grads = []
        for grads_and_vars in zip(*tower_grads):
            grads = tf.stack([g for g, _ in grads_and_vars])
            grad = tf.reduce_mean(grads, 0)
            v = grads_and_vars[0][1]

            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _aggregate_towers(self, corrects, tower_loss_grads):
        """Aggregate the results and gradients over all towers.

        Args:
            corrects: list, a list of the scalars of correct predictions
                for each tower.
            tower_loss_grads: list, a list of gradients for each tower.
        Returns:
            A JoinedResult of summaries, train_op and correct predictions.
        """
        grads = self._average_gradients(tower_loss_grads)
        train_op = self._optimizer.apply_gradients(
            grads, global_step=self._global_step)
        
        summary = tf.summary.merge_all()

        stacked_corrects = tf.stack(corrects)
        summed_corrects = tf.reduce_sum(stacked_corrects, 0)
        accuracy = tf.reduce_mean(tf.cast(stacked_corrects, tf.float32))

        return JoinedResult(summary, train_op, summed_corrects, accuracy)

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
                # Build a tower (replica)
                inferred = self.build_replica()
                # Calculate the loss and predictions
                total_loss, num_correct_per_batch = utils.evaluate(
                    logits=inferred.logits,
                    scope=scope,
                    loss_type=self._hparams.loss_type)
                
                tf.get_variable_scope().reuse_variables()
                total_loss_grad = self._optimizer.compute_gradients(total_loss)

        return TowerResult(inferred, num_correct_per_batch, total_loss_grad)

    def build_model_on_multi_gpus(self, num_gpus):
        """Build the model and Graph and add the train ops on multiple GPUs.

        Divide the inference and gradient computation on multiple GPUs, where
        each GPU has its own replica of the model graph. Then user can get 
        whichever tensor by using `tf.get_tensor_by_name` method.

        Args:
            num_gpus: Number of GPUs available.
        """
        inferreds = []
        corrects = []
        tower_loss_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                tower_out = self._build_single_tower(i)
                inferreds.append(tower_out.inferred)
                corrects.append(tower_out.correct)
                tower_loss_grads.append(tower_out.total_loss_grad)
        
        aggregated_results = self._aggregate_towers(
            corrects, tower_loss_grads)
        return aggregated_results, inferreds