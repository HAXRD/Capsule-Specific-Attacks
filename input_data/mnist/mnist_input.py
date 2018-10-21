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

def inputs(split, data_dir, batch_size, max_epochs):
    """Construct input for mnist experiment.

    Args:
        split: 'train' or 'test', which split of dataset to read from.
        data_dir: path to the mnist tfrecords data directory.
        batch_size: total number of images per batch.
        max_epochs: maximum epochs to go through the model.
    Returns:
        batched_features: a dictionary of the input data features.
    """
    # Dataset specs
    specs = {
        'split': split,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'image_dim': 24, 
        'depth': 3, 
        'num_classes': 10
    }

    if split == 'train':
        pass
    pass