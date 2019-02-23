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

"""Maximizing the difference between one dimension and the rest in one capsule."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pprint import pprint

def compute_grads(tower_idx):
    """Compute the gradients of every dimension - the rest of a specific 
    capsule of the last capsule layer w.r.t. the input tensor.

    Args:
        tower_idx: given tower index, which should be 0 since we are using 
            the first tower in any case.
    Returns:
        grads: the gradients of every dimension - the rest of the most activated capsule
            w.r.t. the input.
        batched_images: placeholder for batched image tensor
        caps_norms_tensor: predicted normalized logits of the model.
    """
    print('{0} Maximizing Difference between Every Dimension and the Rest of Every Capsule {0}'.format('*'*15))
    """Get related tensors"""
    # input batched images tensor
    batched_images = tf.get_collection('tower_%d_batched_images' % tower_idx)[0]
    # visualization related tensors
    visual_tensors = tf.get_collection('tower_%d_visual' % tower_idx)
    # get target tensor 
    caps_out_tensor = visual_tensors[-2] # (?, num_cap_types, num_atoms) (?, 10, 16)
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
    caps_out_name_prefix = '/'.join(caps_out_tensor.name.split('/')[:-1])
    print(caps_out_name_prefix)
    print(caps_out_tensor.get_shape()) # (?, num_cap_types, num_atoms) (?, 10, 16)

    """Split the tensor according to which capsule it is"""
    caps_split_D1_list = tf.split(
        caps_out_tensor, num_or_size_splits=caps_out_tensor.get_shape()[1],
        axis=1, name=caps_out_name_prefix + '/class_split') 
    # [(?, 1, 16), (?, 1, 16), ... x 10]
    
    """Split the tensor according to which dimension it is"""
    caps_split_D2_list = []
    for splited_by_D1_t in caps_split_D1_list:
        # splited_by_D1_t (?, 1, 16)
        temp = tf.split(
            splited_by_D1_t, num_or_size_splits=splited_by_D1_t.get_shape()[2],
            axis=2, name='-'.join(splited_by_D1_t.name.split(':')[:-1]) + '/dim_split')
        # [(?, 1, 1), (?, 1, 1), ... x 16]
        caps_split_D2_list.append(temp)

    """Calculate the dimensional differences"""
    caps_dim_diff_list = []
    for cap_dim_list in caps_split_D2_list:
        cap_dim_sum = tf.reduce_sum(cap_dim_list, axis=2) # (?, 1, 1)
        temp_list = [(16 * cap_dim - cap_dim_sum) / 15
                             for cap_dim in cap_dim_list]
        caps_dim_diff_list.append(temp_list)
    # flatten caps_dim_diff_list
    caps_dim_diff_list = [item for sub in caps_dim_diff_list for item in sub]
    # squeeze out the dimension 2
    caps_dim_diff_list = [tf.squeeze(t, axis=2) for t in caps_dim_diff_list]

    """Compute the gradients"""
    res_grads = []
    for i, caps_single_diff_t in enumerate(caps_dim_diff_list):
        # process name
        caps_single_diff_t_name = "_".join(caps_single_diff_t.name.split(':'))
        # define objective function
        obj_func = caps_single_diff_t
        # compute gradients
        caps_single_diff_grads = tf.gradients(obj_func, batched_images, name='gradients/' + caps_single_diff_t_name)
        # append to resultant list
        res_grads.append(caps_single_diff_grads)
        # print process information
        print('Done processing {0} ---- {1}/{2}      '.format(caps_single_diff_t_name, i+1, len(caps_dim_diff_list)))
    print('')

    """Flatten the list"""
    res_grads = [item for sub in res_grads for item in sub]
    print('Gradients computing completed!')
    
    return res_grads, batched_images, caps_norms_tensor
