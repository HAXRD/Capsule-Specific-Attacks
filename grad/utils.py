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
import numpy as np 
from PIL import Image
import re
import os
import collections

# image analysis
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 

VisualInfo = collections.namedtuple('VisualInfo', ('target_class', 'epoch_idx', 
                                                   'lbl0_class', 'lbl1_class'))       

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
        ga_img_list: a list of 4D tensor with a shape of (1, 1, 24, 24) or (1, 3, 24, 24),
            len(ga_img_list) = {iter_n}
    """
    img = img0.copy() # (1, 1, 24, 24) or (1, 3, 24, 24)

    ga_img_list = [img0.copy()]

    for i in range(iter_n):
        # caculate the gradient values
        g = sess.run(t_grad, feed_dict={in_ph: img})

        # filter out any values that belows the threshold
        g_abs = np.absolute(g)
        filt = np.greater(g_abs, threshold).astype(np.float32)
        g *= filt
        
        # add gradients
        img += g * step # (1, 1, 24, 24) or (1, 3, 24, 24)
        # clip out invalid values
        img = np.clip(img, 0., 1.)

        ga_img_list.append(img) 
    
    return ga_img_list # a list of images with a shape of (1, 1, 24, 24) or (1, 3, 24, 24)

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
    # print(t_grad.name)
    # naive_max_norm gradients/tower_0/logits/split_op_0/tower_0/conv1/Conv2D_grad/Conv2DBackpropInput:0
    # max_norm_diff gradients/tower_0/logits/split_op_0_diff/tower_0/conv1/Conv2D_grad/Conv2DBackpropInput:0
    def _shorten_filename(given_grad_t):
        fn_splitted_list = re.split('/|:', given_grad_t.name)
        # print(fn_splitted_list)
        second_tower_idx = [i for i, part in enumerate(fn_splitted_list) if 'tower' in part][1]
        img_fn = '-'.join(fn_splitted_list[:second_tower_idx])
        return img_fn
    img_fn = _shorten_filename(t_grad) 
    # print(img_fn)
    # naive_max_norm gradients-tower_0-logits-split_op_0
    # max_norm_diff  gradients-tower_0-logits-split_op_0_diff

    # extract useful information for visualization: all string type
    target_class = re.findall(r'\d+', re.findall(r'split_op_\d+', img_fn)[0])[0] # string
    epoch_idx = str(ep_i)
    lbl0_class = str(lbl0)
    lbl1_class = str(lbl1)
    
    visual_info = VisualInfo(target_class, epoch_idx, lbl0_class, lbl1_class)

    # scale up and write to files
    def _write_to_dir(arr, array_type, vis_info, scale_factor, add_base, write_dir, fmt='jpeg'):
        """Create base file name and analysis title description"""
        base_fn = array_type + '-' + \
                  'target_class_' + vis_info.target_class + '-' + \
                  'instance_' + vis_info.epoch_idx + '-' + \
                  'lbl0_' + vis_info.lbl0_class + '-' + \
                  'lbl1_' + vis_info.lbl1_class 
        # title_desc = 'Given input array type: {}\n'.format(array_type) + \
        #              'Label of the target class suppose to maximize: label {}\n'.format(vis_info.target_class) + \
        #              'Instance index of sampled digit: {}th digit\n'.format(vis_info.epoch_idx) + \
        #              'Original image predicted label before processing: label {}\n'.format(vis_info.lbl0_class) + \
        #              'Processed image predicted label: label {}\n'.format(vis_info.lbl1_class)
        title_desc = 'Type = {}; Target Class = {}; Instance Index = {};\n'.format(array_type, vis_info.target_class, vis_info.epoch_idx) + \
                     'Original Prediction = {}; Processed Prediction = {};'.format(vis_info.lbl0_class, vis_info.lbl1_class)

        """Plot 3D surface"""
        assert arr.shape[0] == arr.shape[1]
        arr_size = arr.shape[0]
        x = np.arange(arr_size)
        y = np.arange(arr_size)
        X, Y = np.meshgrid(x, y)
        Z = arr[X, Y]

        fig = plt.figure(figsize=(7, 7))
        fig.suptitle(title_desc)
        
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        # ax.set_title('General')
        ax.contour3D(X, Y, Z, 100, cmap='viridis', alpha=0.5)
        ax.view_init(60, 45)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        # ax.set_title('Top')
        ax.contour3D(X, Y, Z, 100, cmap='viridis', alpha=0.5)
        ax.view_init(90, 0)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        # ax.set_title('Front')
        ax.contour3D(X, Y, Z, 100, cmap='viridis', alpha=0.5)
        ax.view_init(-20, 0)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        # ax.set_title('Side')
        ax.contour3D(X, Y, Z, 100, cmap='viridis', alpha=0.5)
        ax.view_init(-20, 90)

        fig.savefig(os.path.join(write_dir, 'analysis-' + base_fn + '.' + fmt))
        plt.close()

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
        fpath = os.path.join(write_dir, base_fn + '-base_{}-{}x.{}'.format(add_base, scale_factor, fmt))
        # save image
        img.save(fpath, format=fmt)
        # print('Image saved to ', fpath)
    # 1. write original image (no adding base)
    # _write_to_dir(img0, 'img', visual_info, 1, 0.0, write_dir)
    # 2. write original image (add the base of 0.5)
    # _write_to_dir(img0, 'img', visual_info, 1, 0.5, write_dir)
    # 3. write original processed image (add the base of 0.5)
    # _write_to_dir(img1, img_fn + '-img1' + ep_suffix + lbl0_suffix + lbl1_suffix + '-base-' + str(0.5), 1, 0.5, write_dir)
    # _write_to_dir(img1, 'img', visual_info, 1, 0.5, write_dir)
    # 4. write scaled processed image (add the base of 0.5)
    _write_to_dir(img1, 'img', visual_info, 3, 0.5, write_dir)
    # 5. write scaled accumulated gradients (add the base of 0.5)
    _write_to_dir(gsum, 'gsum', visual_info, 3, 0.5, write_dir)
