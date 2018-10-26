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

"""Layer visualization related functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import PIL.Image
from scipy.misc import imresize

import re
import os

import numpy as np 
import tensorflow as tf 
from scipy import ndimage

"""Naive feature visualizations"""
def _write_to_visual_dir(std_img, filename, write_dir, fmt='jpeg'):
    """Saves the normalized images into given directory.

    Args:
        std_img: a normalized image.
        filename: the filename of image.
        write_dir: saving directory.
        fmt: image format.
    """
    arr = np.uint8(np.clip(std_img, 0, 1) * 255) 
    print('image shape: ', std_img.shape)
    f = BytesIO()
    img = PIL.Image.fromarray(arr)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    fpath = os.path.join(write_dir, filename + '.' + fmt)
    fpath = '/'.join([s for s in fpath.split('/') if len(s) > 0])
    img.save(fpath, format=fmt)
    print('Image saved to {}'.format(fpath))

def _stdvisual(img, s=0.1):
    """Normalizes the given image with the shape of (24, 24, 3 or 1)

    Args:
        img: an image with the shape of (1, 3 or 1, 24, 24).
        s: add-on parameter in case the standard = 0.
    Returns:
        img: normalized image.
    """
    return (img - img.mean()) / max(img.std(), 1e-4)*s + 0.5

def _squeeze_transpose(img):
    """Squeeze out the `batch_size` dimension, then transpose into HWC format.

    Args:
        img: an image with the shape of (1, 3 or 1, 24, 24)
    Returns:
        img: a squeezed and transposed image with the shape of (24, 24, 3 or 1)
    """
    img = np.squeeze(img, axis=0)
    img = np.transpose(img, [1, 2, 0])
    return img

def render_naive(t_grad, img0, in_ph_ref, sess, write_dir,
                 iter_n=1000, step=1.0):
    """Naively computes the gradients with given noise image iteratively.

    Args:
        t_grad: the gradient of target objective function w.r.t. the batched
            input placeholder images, actually only 1 image per batch with 
            the shape of (1, 3 or 1, 24, 24) (NCHW)
        img0: the original noise image (1, 3 or 1, 24, 24)
        in_ph_ref: input batched images placeholder, used as the key of feed_dict.
        sess: the running session.
        write_dir: the output directory of the augmented image(s) (after adding 
        gradient values).
        iter_n: number of iterations to add gradients to the noise.
        step: a scalar for each iteration.
    """
    print('gradient shape', t_grad.shape)
    img = img0.copy()
    gsum = np.zeros_like(img)
    
    for i in range(iter_n):
        g = sess.run(t_grad, feed_dict={in_ph_ref: img})
        g /= g.std() + 1e-8

        threshold = 0.5
        g_abs = np.absolute(g)
        filt = np.greater(g_abs, threshold).astype(np.float32)
        g *= filt

        img += g*step
        gsum += g*step
        print('{0:.1f}%'.format((i+1)*100.0/iter_n), end='\r')
    print()
    img = _squeeze_transpose(img)
    
    std_img = _stdvisual(img) 
    std_img = np.squeeze(std_img) # squeeze out the channel dimmension if ch=1
    # shorten filename
    fn_splited_list = re.split('/|:', t_grad.name)
    s_tower_2nd_idx = [i for i, _ in enumerate(fn_splited_list)][1]
    std_img_fn = '-'.join(fn_splited_list[:s_tower_2nd_idx])
    
    _write_to_visual_dir(std_img, std_img_fn, write_dir)

    if len(std_img.shape) == 3:
        scale_list = [5.0, 5.0, 1.0]
    else:
        scale_list = [5.0, 5.0]

    scaled_img = ndimage.zoom(std_img, scale_list)
    scaled_img_fn = '5x-' + std_img_fn
    _write_to_visual_dir(scaled_img, scaled_img_fn, write_dir)

    gsum = _squeeze_transpose(gsum)
    std_gsum = _stdvisual(gsum)
    std_gsum = np.squeeze(std_gsum)
    std_gsum_fn = 'gsum-' + std_img_fn
    _write_to_visual_dir(std_gsum, std_gsum_fn, write_dir)

    scaled_gsum = ndimage.zoom(std_gsum, scale_list)
    scaled_gsum_fn = '5x-' + std_gsum_fn
    _write_to_visual_dir(scaled_gsum, scaled_gsum_fn, write_dir)


"""Multiscale feature visualizations"""
def _cal_grad_tiled(img, t_grad, in_ph_ref, sess, tile_size=24):
    """Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blr tile boundaries over 
    multiple iterations.

    Args:
        img: shape (3 or 1, 48, 48)
    Returns:
        shape (1, 3 or 1, 48, 48)
    """
    img = np.expand_dims(img, axis=0) # (1, 48, 48, 3 or 1)
    img = np.transpose(img, [0, 3, 1, 2]) # (1, 3 or 1, 48, 48)

    sz = tile_size 
    h, w = img.shape[-2:] # [48, 48]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 3), sy, 2)
    grad = np.zeros_like(img)

    for y in range(0, max(h - sz//2, sz), sz):
        for x in range(0, max(w - sz//2, sz), sz):
            sub = img_shift[:, :, y:y+sz, x:x+sz]
            g = sess.run(t_grad, {in_ph_ref: sub})
            grad[:, :, y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 3), -sy, 2)

def _resize(img, size):
    """Resize the image using bilinear iterpolation
    Args:
        img: (24, 24, 3 or 1)
        size: [48, 48]
    Returns:
        resized image (48, 48, 3 or 1)
    """
    img = np.expand_dims(img, 0)
    img_t = tf.placeholder(dtype=tf.float32)
    scaled_img_t = tf.image.resize_bilinear(img_t, size)
    scaled_img = scaled_img_t.eval({img_t: img})[0, :, :, :]
    return scaled_img

def render_multiscale(t_grad, img0, in_ph_ref, sess, write_dir,
                      iter_n=100, step=1.0, octave_n=3, octave_scale=2.0):
    """Perform the feature visualizations on images and scale up the size
    
    Args:
        t_grad: the gradient of target objective function w.r.t. the batched
            input placeholder images, actually only 1 image per batch with 
            the shape of (1, 3 or 1, 24, 24) (NCHW)
        img0: the original noise image (1, 3 or 1, 24, 24)
        in_ph_ref: input batched images placeholder, used as the key of feed_dict.
        sess: the running session.
        write_dir: the output directory of the augmented image(s) (after adding 
        gradient values).
        iter_n: number of iterations to add gradients to the noise.
        step: a scalar for each iteration.
        octave_n: the number of times to scale the output image.
        octave_scale: the scale value for each scale.
    """
    img = img0.copy() # (1, 3 or 1, 24, 24)
    img = _squeeze_transpose(img) # (24, 24, 3 or 1)

    for octave in range(octave_n):
        if octave > 0:
            hw = np.float32(img.shape[:2]) * octave_scale # [48., 48.]
            img = _resize(img, np.int32(hw)) # (48, 48, 3 or 1)
            # img = imresize(img, hw) / 255.
        for i in range(iter_n):
            g = _cal_grad_tiled(img, t_grad, in_ph_ref, sess) # (1, 3 or 1, 48, 48)
            g /= g.std() + 1e-8
            g = _squeeze_transpose(g) # (48, 48, 3 or 1)
            img += g*step

        std_img = _stdvisual(img)
        std_img = np.squeeze(std_img) # squeeze out the channel dimmension if ch=1
        std_img_fn = std_img_fn = '-'.join(re.split('/|:', t_grad.name)) + '-octave{}'.format(str(octave))
        _write_to_visual_dir(std_img, std_img_fn, write_dir)

        if len(std_img.shape) == 3:
            scale_list = [2.0, 2.0, 1.0]
        else:
            scale_list = [2.0, 2.0]
        scaled_img = ndimage.zoom(std_img, scale_list)
        scaled_img_fn = '2x-' + std_img_fn
        _write_to_visual_dir(scaled_img, scaled_img_fn, write_dir)

"""Laplacian Pyramid Gradient Normalization"""
def render_lapnorm(t_grad, img0, in_ph_ref, sess, write_dir,
                   iter_n=8, step=1.0, octave_n=5, octave_scale=2.0, lap_n=4):
    pass
