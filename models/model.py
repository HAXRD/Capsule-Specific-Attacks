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

import abc 
import collections
import tensorflow as tf 
from models.layers import utils

Inferred = collections.namedtuple('Inferred',
                                 ('logits', 'remakes'))
TowerResult = collections.namedtuple('TowerResult',
                                    ('inferred', 'correct', 'grads'))
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

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each variable across all towers.

        Args:
            tower_grads: a list of gradient lists for each tower. Each gradient list
                is a list of (gradient, variable) tuples for all variables.
        Returns:
            A list of pairs of (gradient, variable) where gradient has been averaged 
            across all towers.
        """
        averaged_grads = []
        for grads_and_vars in zip(*tower_grads):
            grads = tf.stack([g for g, _ in grads_and_vars])
            grad = tf.reduce_mean(grads, 0)

            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            averaged_grads.append(grad_and_var)
        
        return averaged_grads

    def _join_tower_results(self, corrects, tower_grads):
        """Aggregates the results and gradients over all towers.

        Args:
            corrects: a list of the numbers of correct predictions for each tower.
            tower_grads: a list of gradient lists for each tower.
        Returns:
            A JoinedResult of evaluation results.
        """
        # average gradients
        grads = self._average_gradients(tower_grads)
        # apply gradients
        train_op = self._optimizer.apply_gradients(
            grads, global_step=self._global_step)
        # add summaries
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary = tf.summary.merge(summaries)
        # stack corrects
        stacked_corrects = tf.stack(corrects)
        # sum up the correct predictions
        summed_corrects = tf.reduce_sum(stacked_corrects, 0)
        # calculate overall accuracy
        accuracy = tf.reduce_mean(stacked_corrects, 0)
        
        return JoinedResult(summary, train_op, summed_corrects, accuracy)

    @abc.abstractmethod
    def build_replica(self, tower_idx):
        """Adds a replica graph ops.

        Builds the architecture of the neural net to derive logits from 
        batched_dataset. The inference graph defined here should involve 
        trainable variables otherwise the optimizer will raise a ValueError.

        Args:
            tower_idx: the index number for this tower. Each tower is named
                as tower_{tower_idx} and resides on gpu:{tower_idx}.
        Returns:
            Inferred namedtuple containing (logits, None).
        """
        raise NotImplementedError('Not implemented.')

    def _build_single_tower(self, tower_idx):
        """Calculates the model gradient for one tower.
        
        Adds the inference and loss operations to the graph. Calculates the 
        gradients based on the loss. 

        Args:
            tower_idx: the index number for this tower. Each tower is named
                as tower_{tower_idx} and resides on gpu:{tower_idx}.
        Returns:
            TowerResult: a namedtuple containing inferred logits, number of correct
                predictions per batch, and the gradients 
        """
        with tf.device('/gpu:%d' % tower_idx):
            with tf.name_scope('tower_%d' % tower_idx) as scope:
                # build a tower/replica
                inferred = self.build_replica(tower_idx)
                # calculate the loss and number of correct predictions per batch
                total_loss, num_correct_per_batch = utils.evaluate(
                    logits=inferred.logits,
                    scope=scope,
                    loss_type=self._hparams.loss_type)
                # reuse variables
                tf.get_variable_scope().reuse_variables()
                grads = self._optimizer.compute_gradients(total_loss)
        
        return TowerResult(inferred, num_correct_per_batch, grads)

    def build_model_on_multi_gpus(self):
        """Build the model and Graph and add the train ops on single GPUs.

        Divides the inference and gradient computation on multiple GPUs.
        The aggregates the gradients and return the resultant ops.

        Returns:
            joined_results: a namedtuple containing 
        """
        inferreds = []
        corrects = []
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self._specs['num_gpus']):
                # build single tower
                tower_output = self._build_single_tower(i)
                # append to lists
                inferreds.append(tower_output.inferred)
                corrects.append(tower_output.correct)
                tower_grads.append(tower_output.grads)
        
        # join the results of towers
        joined_result = self._join_tower_results(corrects, tower_grads)

        tf.add_to_collection('summary', joined_result.summary)
        tf.add_to_collection('train_op', joined_result.train_op)
        tf.add_to_collection('correct', joined_result.correct)
        tf.add_to_collection('accuracy', joined_result.accuracy)

        return joined_result
        
