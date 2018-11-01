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

"""Naively maximizing the norm of one specific capsule of the last layer."""

import tensorflow as tf

def compute_grads(tower_idx):
    """Compute the gradients of the logit norms of the last capsule layer
    w.r.t. the input tensor.

    Args:
        tower_idx: given tower index, which should be 0 since we are using 
            the first tower in any case.
    Returns:
        grads: the gradients of capsule norms w.r.t. the input.
    """

    """Get related tensors"""
    # input batched images tensor
    batched_images = tf.get_collection('tower_%d_batched_images' % tower_idx)[0]
    # visualization related tensors
    visual_tensors = tf.get_collection('tower_%d_visual' % tower_idx)
    # get target tensor 
    caps_norms_tensor = visual_tensors[-1] # (?, num_cap_types) (?, 10)

    """Report the tensor information"""
    print('='*10)
    for i, vt in enumerate(visual_tensors):
        if i == len(visual_tensors) - 1:
            print('visual tensor name: {} (target)'.format(vt.name))
        else:
            print('visual tensor name: {}'.format(vt.name))
    print('='*10)


    # shorten tensor prefix name
    caps_norms_name_prefix = '/'.join(caps_norms_tensor.name.split('/')[:-1]) + '/' \
                           + caps_norms_tensor.name.split('/')[-1][:-2]
    print(caps_norms_name_prefix)
    print(caps_norms_tensor.get_shape())

    """Split the norms into individual norm"""
    caps_norm_list = tf.split(
        caps_norms_tensor, num_or_size_splits=caps_norms_tensor.get_shape()[1],
        axis=1, name=caps_norms_name_prefix + '/split_op')
    
    """Compute norm gradients"""
    res_grads = []
    for i, caps_norm_t in enumerate(caps_norm_list):
        # process name
        caps_norm_t_name = '_'.join(caps_norm_t.name.split(':'))
        # define objective function
        obj_func = caps_norm_t
        # compute gradients
        caps_norm_grads = tf.gradients(obj_func, batched_images, name='gradients/' + caps_norm_t_name)
        # append to resultant list
        res_grads.append(caps_norm_grads)
        # print process information
        print('Done processing {0} ---- {1:.2f}%'.format(
            caps_norm_t_name, (1+i)*100.0/float(len(caps_norm_t))))
    print('')

    print('Gradients computing completed!')
    
    return res_grads
