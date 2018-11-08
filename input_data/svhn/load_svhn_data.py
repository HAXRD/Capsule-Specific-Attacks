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
import os
import numpy as np 
from scipy.io import loadmat

def load_svhn(path, split='train'):
    """Load svhn dataset from mat files.

    Args:
        path: given directory of where the dataset was stored
        split: 'train' (73257) or 'test' (26032)
    Returns: 
        images: numpy array storing image data, uint8 0 ~ 255, (?, 32, 32, 3)
        labels: numpy array storing label data, uint8 0 ~ 9, (?, 1)
    """
    data = os.path.join(
        path, '%s_32x32.mat' % split)
    
    # read data
    mat = loadmat(data) # 'X' (32, 32, 3, 73257)
                        # 'y' (73257, 1)
    
    images = np.transpose(mat['X'], [3, 0, 1, 2])
    labels = np.reshape(mat['y'], -1) - 1

    # images: (?, 32, 32, 3)
    # labels: (?, )
    return images, labels



    
