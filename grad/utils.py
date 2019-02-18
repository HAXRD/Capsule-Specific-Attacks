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

def run_gradient_ascent(t_grad, img0, in_ph, sess,
                        iter_n, step, threshold=0.0):
    """Run gradient ascent to the given image and only record those results at 
    iter_ns_to_record = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 20, 40, 60, 80, 
                         100, 200, 400, 600, 800, 1000]

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
        iter_n_recorded: iterations number recorded
        ga_img_list: a list of images, where images are 4D tensors with the 
        shape of (1, 1, 24, 24) or (1, 3, 24, 24),
    """
    assert iter_n >= 10
    iter_ns_to_record = [1, 2, 3, 4, 5, 
                         10, 20, 40, 60, 80]

    img = img0.copy() # (1, 1, 24, 24) or (1, 3, 24, 24)

    ga_img_list = [img0.copy()]
    iter_n_recorded = [0]

    for i in range(1, iter_n + 1):
        # caculate the gradient values
        g = sess.run(t_grad, feed_dict={in_ph: img})

        # fgsm
        g = np.sign(g)
        
        # add gradients
        img += g * step # (1, 1, 24, 24) or (1, 3, 24, 24)
        # clip out invalid values
        img = np.clip(img, 0., 1.)

        # record results
        if i in iter_ns_to_record:
            ga_img_list.append(img)
            iter_n_recorded.append(i)
    
    return iter_n_recorded, ga_img_list # a list of (idex, image), where images have the shape of (1, 1, 24, 24) or (1, 3, 24, 24)
