# Credit https://github.com/zalandoresearch/fashion-mnist#get-the-data
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
import gzip
import numpy as np

def load_fashion_mnist(path, split='train'):
    """Load fashion-mnist dataset from byte files.

    Args:
        path: given directory of where the dataset was stored
        split: 'train' or 'test'.
    Return:
        images: numpy array storing image data, uint8 0 ~ 255, (60000 or 10000, 28, 28)
        labels: numpy array storing label data, uint8 0 ~ 9, (60000 or 10000,)
    """
    if split == 'test':
        split = 't10k'
    labels_path = os.path.join(
        path, '%s-labels-idx1-ubyte.gz' % split)
    images_path = os.path.join(
        path, '%s-images-idx3-ubyte.gz' % split)

    # read label data    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(
            lbpath.read(), dtype=np.uint8, offset=8)
    # read image data
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels 