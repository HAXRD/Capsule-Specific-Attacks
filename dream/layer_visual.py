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

import re
import os

import numpy as np 
import tensorflow as tf 

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
    f = BytesIO()
    img = PIL.Image.fromarray(arr)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    fpath = os.path.join(write_dir, filename + '.' + fmt)
    img.save(fpath, format=fmt)
    print('Image saved to {}'.format(fpath))

def _stdvisual(img, s=0.1):
    """Normalizes the given image with the shape of (32, 32, 3)

    Args:
        img: an image with the shape of (1, 3, 32, 32).
        s: add-on parameter in case the standard = 0.
    Returns:
        img: normalized image.
    """
    return (img - img.mean()) / max(img.std(), 1e-4)*s + 0.5

def _squeeze_transpose(img):
    """Squeeze out the `batch_size` dimension, then transpose into HWC format.

    Args:
        img: an image with the shape of (1, 3, 32, 32)
    Returns:
        img: a squeezed and transposed image with the shape of (32, 32, 3)
    """
    img = np.squeeze(img, axis=0)
    img = np.transpose(img, [1, 2, 0])
    return img

def render_naive(t_grad, img0, in_ph_ref, sess, write_dir,
                 iter_n=20, step=1.0):
    """Naively computes the gradients with given noise image iteratively.

    Args:
        t_grad: the gradient of target objective function w.r.t. the batched
            input placeholder images, actually only 1 image per batch with 
            the shape of (1, 3, 32, 32) (NCHW)
        img0: the original noise image (1, 3, 32, 32)
        in_ph_ref: input batched images placeholder, used as the key of feed_dict.
        sess: the running session.
        write_dir: the output directory of the augmented image(s) (after adding 
        gradient values).
        iter_n: number of iterations to add gradients to the noise.
        step: a scalar for each iteration.
    """
    img = img0.copy()
    for i in range(iter_n):
        g = sess.run(t_grad, feed_dict={in_ph_ref: img})
        g /= g.std() + 1e-8
        img += g*step
    
    img = _squeeze_transpose(img)
    std_img = _stdvisual(img)
    std_img_fn = '-'.join(re.split('/|:', t_grad.name))
    write_dir += '/naive/'

    _write_to_visual_dir(std_img, std_img_fn, write_dir)

"""Multiscale feature visualizations"""
def _cal_grad_tiled(img, t_grad, in_ph_ref, sess, tile_size=512):
    """Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blr tile boundaries over 
    multiple iterations.

    Args:
        img: shape (3, 64, 64)
    """
    img = np.expand_dims(img, axis=0) # (1, 64, 64, 3)
    img = np.transpose(img, [0, 3, 1, 2]) # (1, 3, 64, 64)

    sz = tile_size 
    h, w = img.shape[-2:] # [64, 64]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 3), sy, 2)
    grad = np.zeros_like(img)

    for y in range(0, max(h - sz//2, sz), sz):
        for x in range(0, max(w - sz//2, sz), sz):
            sub = img_shift[:, :, y:y+sz, x:x+sz]
            g = sess.run(t_grad, {in_ph_ref: sub})
            grad[:, :, y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 3), -sy, 2)

def tffunc(*argtypes):
    """Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    """
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def _resize(img, size):
    """Resize the image using bilinear iterpolation
    Args:
        img: (32, 32, 3)
        size: [64, 64]
    Returns:
        resized image (64, 64, 3)
    """
    scaled_img_t = tf.image.resize_bilinear(img, size)[0, :, :, :] # (64, 64, 3)
    return scaled_img_t
_resize = tffunc(np.float32, np.int32)(_resize)

def render_multiscale(t_grad, img0, in_ph_ref, sess, write_dir,
                      iter_n=10, step=1.0, octave_n=3, octave_scale=2.0):
    """Perform the feature visualizations on images and scale up the size
    
    Args:
        t_grad: the gradient of target objective function w.r.t. the batched
            input placeholder images, actually only 1 image per batch with 
            the shape of (1, 3, 32, 32) (NCHW)
        img0: the original noise image (1, 3, 32, 32)
        in_ph_ref: input batched images placeholder, used as the key of feed_dict.
        sess: the running session.
        write_dir: the output directory of the augmented image(s) (after adding 
        gradient values).
        iter_n: number of iterations to add gradients to the noise.
        step: a scalar for each iteration.
        octave_n: the number of times to scale the output image.
        octave_scale: the scale value for each scale.
    """
    img = img0.copy() # (1, 3, 32, 32)
    img = _squeeze_transpose(img) # (32, 32, 3)

    for octave in range(octave_n):
        if octave > 0:
            hw = np.float32(img.shape[:2]) * octave_scale # [64., 64.]
            img = _resize(img, np.int32(hw)) 
        for i in range(iter_n):
            g = _cal_grad_tiled(img, t_grad, in_ph_ref, sess)
            g /= g.std() + 1e-8
            img += g*step

        std_img = _stdvisual(img)
        std_img_fn = std_img_fn = '-'.join(re.split('/|:', t_grad.name)) + '-octave{}'.format(str(octave))
        write_dir += '/multiscale/'
        _write_to_visual_dir(std_img, std_img_fn, write_dir)

    
        