# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Library for capsule layers.

This has the layer implementation for coincidence detection, routing and 
capsule layers.
"""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

from models.layers import variables

def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin
    is 0.1 and for positives it is 0.9. First subtract 0.5 from all logits.
    Now margin is 0.4 from each side.

    Args:
        labels: tensor, one hot encoding of ground truth.
        raw_logits: tensor, model predictions in range [0, 1]
        margin: scalar, the margin after subtracting .5 from raw_logits.
        downweight: scalar, the factor for negative cost.
    """
    logits = raw_logits - .5
    positive_cost = labels * tf.cast(tf.less(logits, margin), tf.float32) \
                  * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(tf.greater(logits, -margin), tf.float32) \
                  * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost

def evaluate(logits, labels, num_targets, scope, loss_type):
    """Calculates total loss and performance metrics like accuracy.

    Args:
        logits: tensor, output of the model
        labels: tensor, ground truth of the data.
        num_targets: scalar, number of present objects in image,
                     i.e. the number of 1s in labels.
        scope: The scope to collect losses of.
        loss_type: 'sigmoid' (num_targets > 1), 'softmax' or 'margin' for 
                    margin loss.
    Returns:
        The total loss of the model, number of correct predictions and number of
        cases where at least one of the classes is correctly predicted.
    Raises:
        NotImplementedError: if the loss_type is not sigmoid, softmax or margin loss.
    """
    with tf.name_scope('loss'):
        if loss_type == 'sigmoid':
            classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels/2.0, logits=logits)
        elif loss_type == 'softmax':
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        elif loss_type == 'margin':
            classification_loss = _margin_loss(labels=labels, raw_logits=logits)
        else:
            raise NotImplementedError('Not implemented!')
        
        with tf.name_scope('total'):
            batch_classification_loss = tf.reduce_mean(classification_loss) 
            tf.add_to_collection('losses', batch_classification_loss)
    tf.summary.scalar('batch_classification_loss', batch_classification_loss)

    all_losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(all_losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            _, targets = tf.nn.top_k(labels, k=num_targets)
            _, predictions = tf.nn.top_k(logits, k=num_targets)
            missed_targets = tf.contrib.metrics.set_difference(targets, predictions)
            num_missed_targets = tf.contrib.metrics.set_size(missed_targets)
            correct = tf.equal(num_missed_targets, 0)
            almost_correct = tf.less(num_missed_targets, num_targets)
            correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
            almost_correct_sum = tf.reduce_sum(tf.cast(almost_correct, tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('correct_prediction_batch', correct_sum)
    tf.summary.scalar('almost_correct_batch', almost_correct_sum)

    return total_loss, correct_sum, almost_correct_sum