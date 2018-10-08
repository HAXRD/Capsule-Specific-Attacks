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


def get_gradients():
    """Compute all the gradients for each variable w.r.t. the input.

    Args:
        logits: 
    Return:

    """
    graph = tf.get_default_graph()
    all_op_names = [n.name for n in graph.as_graph_def().node]
    print(all_op_names)
    """
        ['per_image_standardization/image', 'per_image_standardization/control_dependency', 'per_image_standardization/Shape', 'per_image_standardization/Const', 'per_image_standardization/Prod', 'per_image_standardization/Cast', 'per_image_standardization/Const_1', 'per_image_standardization/Mean', 'per_image_standardization/Square', 'per_image_standardization/Const_2', 'per_image_standardization/Mean_1', 'per_image_standardization/Square_1', 'per_image_standardization/sub', 'per_image_standardization/Relu', 'per_image_standardization/Sqrt', 'per_image_standardization/Cast_1', 'per_image_standardization/Rsqrt', 'per_image_standardization/Maximum', 'per_image_standardization/Sub', 'per_image_standardization', 'transpose/perm', 'transpose', 'batch/Const', 'batch/fifo_queue', 'batch/fifo_queue_enqueue', 'batch/fifo_queue_Close', 'batch/fifo_queue_Close_1', 'batch/fifo_queue_Size', 'batch/ToFloat', 'batch/mul/y', 'batch/mul', 'batch/fraction_of_103_full/tags', 'batch/fraction_of_103_full', 'batch/n', 'batch', 'global_step/Initializer/Const', 'global_step', 'global_step/Assign', 'global_step/read', 'ExponentialDecay/learning_rate', 'ExponentialDecay/Cast_1/x', 'ExponentialDecay/Cast_1', 'ExponentialDecay/Cast_2/x', 'ExponentialDecay/truediv', 'ExponentialDecay/Pow', 'ExponentialDecay', 'Maximum/y', 'Maximum', 'Reshape/shape', 'Reshape', 'conv1/weights_1/Initializer/truncated_normal/shape', 'conv1/weights_1/Initializer/truncated_normal/mean', 'conv1/weights_1/Initializer/truncated_normal/stddev', 'conv1/weights_1/Initializer/truncated_normal/TruncatedNormal', 'conv1/weights_1/Initializer/truncated_normal/mul', 'conv1/weights_1/Initializer/truncated_normal', 'conv1/weights_1', 'conv1/weights_1/Assign', 'conv1/weights_1/read', 'conv1/Conv2D', 'conv1/biases/Initializer/Const', 'conv1/biases', 'conv1/biases/Assign', 'conv1/biases/read', 'conv1/BiasAdd', 'conv1/conv1', 'conv1/MaxPool2D/MaxPool', 'conv2/weights_1/Initializer/truncated_normal/shape', 'conv2/weights_1/Initializer/truncated_normal/mean', 'conv2/weights_1/Initializer/truncated_normal/stddev', 'conv2/weights_1/Initializer/truncated_normal/TruncatedNormal', 'conv2/weights_1/Initializer/truncated_normal/mul', 'conv2/weights_1/Initializer/truncated_normal', 'conv2/weights_1', 'conv2/weights_1/Assign', 'conv2/weights_1/read', 'conv2/Conv2D', 'conv2/biases/Initializer/Const', 'conv2/biases', 'conv2/biases/Assign', 'conv2/biases/read', 'conv2/BiasAdd', 'conv2/conv2', 'conv2/MaxPool2D/MaxPool', 'Flatten/flatten/Shape', 'Flatten/flatten/strided_slice/stack', 'Flatten/flatten/strided_slice/stack_1', 'Flatten/flatten/strided_slice/stack_2', 'Flatten/flatten/strided_slice', 'Flatten/flatten/Reshape/shape/1', 'Flatten/flatten/Reshape/shape', 'Flatten/flatten/Reshape', 'fc1/weights_1/Initializer/truncated_normal/shape', 'fc1/weights_1/Initializer/truncated_normal/mean', 'fc1/weights_1/Initializer/truncated_normal/stddev', 'fc1/weights_1/Initializer/truncated_normal/TruncatedNormal', 'fc1/weights_1/Initializer/truncated_normal/mul', 'fc1/weights_1/Initializer/truncated_normal', 'fc1/weights_1', 'fc1/weights_1/Assign', 'fc1/weights_1/read', 'fc1/biases/Initializer/Const', 'fc1/biases', 'fc1/biases/Assign', 'fc1/biases/read', 'fc1/MatMul', 'fc1/add', 'fc1/fc1', 'softmax_layer/weights_1/Initializer/truncated_normal/shape', 'softmax_layer/weights_1/Initializer/truncated_normal/mean', 'softmax_layer/weights_1/Initializer/truncated_normal/stddev', 'softmax_layer/weights_1/Initializer/truncated_normal/TruncatedNormal', 'softmax_layer/weights_1/Initializer/truncated_normal/mul', 'softmax_layer/weights_1/Initializer/truncated_normal', 'softmax_layer/weights_1', 'softmax_layer/weights_1/Assign', 'softmax_layer/weights_1/read', 'softmax_layer/biases/Initializer/Const', 'softmax_layer/biases', 'softmax_layer/biases/Assign', 'softmax_layer/biases/read', 'softmax_layer/MatMul', 'softmax_layer/logits']
    """
    input_images = graph.get_tensor_by_name('image_4d:0')
    logits = graph.get_tensor_by_name('softmax_layer/logits:0')
    print(logits.shape)
    logits_after_softmax = tf.nn.softmax(logits, name='logits_after_softmax')
    logits_after_softmax_max = tf.reduce_max(logits_after_softmax, name='max_act_softmax')
    logits_grad = tf.gradients(logits_after_softmax, input_images)[0]
    print(logits_grad.shape)
    return logits_grad, None



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