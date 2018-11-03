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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
from PIL import Image
import re
import os

def compute_grads(tower_idx):
    """Compute the gradients of the logit norms of the last capsule layer
    w.r.t. the input tensor.

    Args:
        tower_idx: given tower index, which should be 0 since we are using 
            the first tower in any case.
    Returns:
        grads: the gradients of capsule norms w.r.t. the input.
        batched_images: placeholder for batched image tensor
        caps_norms_tensor: predicted normalized logits of the model.
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
            caps_norm_t_name, (1+i)*100.0/float(len(caps_norm_list))))
    print('')

    """Flatten the list"""
    res_grads = [item for sub in res_grads for item in sub]
    print('Gradients computing completed!')
    
    return res_grads, batched_images, caps_norms_tensor

def run_gradient_ascent(t_grad, img0, in_ph, sess,
                        iter_n, step, threshold=0.0):
    """

    Args:
        t_grad: the gradients of target objective function w.r.t. the batched
            input placeholder images, but actually there is only 1 image per
            batch with the shape of (1, 1, 24, 24) or (1, 3, 24, 24) (NCHW)
        img0: the original batched input images, (1, 1, 24, 24) or (1, 3, 24, 24) (NCHW)
        in_ph: input batched image placeholder, used as the key of feed dict.
        sess: the running session.
        iter_n: number of iterations to add gradients to the img0.
        step: step size multiplier of each iteration.
        threshold: gradient lower bound threshold, any calculated gradients under this
            value will be ignored.
    Returns:
        img: the resultant gradient ascended image, (1, 1, 24, 24) or (1, 3, 24, 24)
        gsum: the accumulated gradient sum, (1, 1, 24, 24) or (1, 3, 24, 24)
    """
    # show gradient tensor information
    print('gradient shape ~ ', t_grad.shape)
    img = img0.copy() # (1, 1, 24, 24) or (1, 3, 24, 24)
    gsum = np.zeros_like(img) # all 0, TODO: need to handle the negative gradients.

    for i in range(iter_n):
        # calculate the gradient values
        g = sess.run(t_grad, feed_dict={in_ph: img})

        # filter out any values that belows the threshold
        g_abs = np.absolute(g)
        filt = np.greater(g_abs, threshold).astype(np.float32)
        g *= filt

        img += g*step # (1, 1, 24, 24) or (1, 3, 24, 24)
        gsum += g*step # (1, 1, 24, 24) or (1, 3, 24, 24)

        print('{0:.1f}%'.format((i+1)*100.0/iter_n), end='\r')
    print()

    return img, gsum

def write_results(write_dir, t_grad, gsum, img0, img1, lbl0, lbl1, ep_i):
    """
    Args:
        write_dir: output directory to store the data.
        t_grad: the gradients of target objective function w.r.t. the batched
            input placeholder images, but actually there is only 1 image per
            batch with the shape of (1, 1, 24, 24) or (1, 3, 24, 24) (NCHW)
        img0: unprocessed image, (1, 1, 24, 24) or (1, 3, 24, 24)
        img1: processed image, (1, 1, 24, 24) or (1, 3, 24, 24)
        lbl0: predicted label of unprocessed image.
        lbl1: predicted label of processed image.
        ep_i: the index of current epoch.
    """
    # transpose and squeeze out dimensions equal to 1
    def _transpose_n_squeeze(given_img):
        given_img = np.transpose(given_img, [0, 2, 3, 1]) # (1, 24, 24, 1) or (1, 24, 24, 3)
        given_img = np.squeeze(given_img) # (24, 24) or (24, 24, 3)
        return given_img
    gsum = _transpose_n_squeeze(gsum)
    img0 = _transpose_n_squeeze(img0)
    img1 = _transpose_n_squeeze(img1)
    assert img0.shape == img1.shape

    # shorten the filename
    def _shorten_filename(given_grad_t):
        fn_splitted_list = re.split('/|:', given_grad_t.name)
        second_tower_idx = [i for i, part in enumerate(fn_splitted_list) if 'tower' in part][1]
        img_fn = '-'.join(fn_splitted_list[:second_tower_idx])
        return img_fn
    img_fn = _shorten_filename(t_grad)

    # create epoch prefix, label0 prefix and label1 prefix
    ep_suffix = '-ep-' + str(ep_i)
    lbl0_suffix = '-lbl0-' + str(lbl0)
    lbl1_suffix = '-lbl1-' + str(lbl1)

    # scale up and write to files
    def _write_to_dir(arr, fn, scale_factor, add_base, write_dir, fmt='jpeg'):
        """Process image"""
        # add base to the array values, suppose add_base=0.5
        arr += add_base # the the scale changes from 0. ~ 1. to 0. ~ 2.
        # normalize back to 0. ~ 1.
        arr /= (1. + 2 * add_base)
        # convert to 0 ~ 255 uint8
        arr_uint8 = np.uint8(arr * 255.)
        # clip out exceeding values
        arr_uint8 = np.clip(arr_uint8, 0, 255)
        
        """Save image"""
        # get image mode
        if len(arr_uint8.shape) == 3:
            mode = 'RGB'
        elif len(arr_uint8.shape) == 2:
            mode = 'L'
        # convert into Image object
        img = Image.fromarray(arr_uint8, mode)
        # scale up the image
        assert type(scale_factor) is int
        scaled_size = (arr.shape[0] * scale_factor, arr.shape[1] * scale_factor)
        img = img.resize(scaled_size, resample=Image.BILINEAR)
        # create directory if not exists
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        fpath = os.path.join(write_dir, fn + '.' + fmt)
        # save image
        img.save(fpath, format=fmt)
        print('Image saved to ', fpath)
    # 1. write original image (no adding base)
    # _write_to_dir(img0, img_fn + '-img0' + ep_suffix + lbl0_suffix, 1, 0.0, write_dir)
    # 2. write original image (add the base of 0.5)
    # _write_to_dir(img0, img_fn + '-img0' + ep_suffix + lbl0_suffix + '-base-' + str(0.5), 1, 0.5, write_dir)
    # 3. write original processed image (add the base of 0.5)
    # _write_to_dir(img1, img_fn + '-img1' + ep_suffix + lbl0_suffix + lbl1_suffix + '-base-' + str(0.5), 1, 0.5, write_dir)
    # 4. write scaled processed image (add the base of 0.5)
    _write_to_dir(img1, img_fn + '-img1' + ep_suffix + lbl0_suffix + lbl1_suffix + '-base-' + str(0.5) + '-3x', 3, 0.5, write_dir)
    # 5. write scaled accumulated gradients (add the base of 0.5)
    _write_to_dir(gsum, img_fn + '-gsum' + ep_suffix + lbl0_suffix + lbl1_suffix + '-base-' + str(0.5) + '-3x', 3, 0.5, write_dir)
    
