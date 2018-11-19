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
from glob import glob 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch

"""
.                           <--- data_dir
├── caps_full               <--- model_lvl_dir
│   ├── cifar10             <--- dataset_lvl_dir
│   │   ├── evaluate        <--- load_dir
│   │   ├── max_caps_dim_diff
│   │   ├── max_norm_diff
│   │   ├── naive_max_caps_dim
│   │   ├── naive_max_norm
│   │   ├── noise_max_caps_dim_diff
│   │   ├── noise_max_norm_diff
│   │   ├── noise_naive_max_caps_dim
│   │   └── noise_naive_max_norm
│   ├── fmnist
│   ├── mnist
│   └── svhn
└── cnn
    ├── cifar10
    │   ├── evaluate
    │   ├── max_norm_diff
    │   ├── naive_max_norm
    │   ├── noise_max_norm_diff
    │   └── noise_naive_max_norm
    ├── fmnist
    ├── mnist
    └── svhn
"""

"""Dataset classes"""
MNIST_CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
FMNIST_CATEGORIES = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
SVHN_CATEGORIES = MNIST_CATEGORIES
CIFAR10_CATEGORIES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

categories = {
    'mnist': MNIST_CATEGORIES,
    'fmnist': FMNIST_CATEGORIES,
    'svhn': SVHN_CATEGORIES,
    'cifar10': CIFAR10_CATEGORIES
}

"""Define CONSTANTS"""
AVAILABLE_ITER_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 20, 40, 60, 80,
                     100, 200, 400, 600, 800, 1000]
NORM_ASPECT_METHODS = ['naive_max_norm', 'max_norm_diff', 'noise_naive_max_norm', 'noise_max_norm_diff']
DIRECTION_ASPECT_METHODS = ['naive_max_caps_dim', 'max_norm_diff', 'noise_naive_max_caps_dim', 'noise_max_norm_diff']

"""Set up data directories"""
data_dir = '/Users/xu/Storage/final'

"""Utility functions"""
def get_model_lvl_dirs(data_dir, model_pattern='*'):
    """Get model level directories.
    
    Args:
        data_dir: directory to data;
        model_pattern: patterns to match the model, '*' find all the models.
    Returns:
        model_lvl_dirs: a list, model level directories. 
    """
    return sorted(glob(os.path.join(data_dir, model_pattern)))

def get_dataset_lvl_dirs(model_lvl_dir, dataset_pattern='*'):
    """Get dataset level directories.
    
    Args:
        model_lvl_dir: model level directory;
        dataset_pattern: patterns to match the dataset, '*' find all the datasets.
    Returns:
        dataset_dirs: a list, dataset level directories.
    """
    return sorted(glob(os.path.join(model_lvl_dir, dataset_pattern)), key=lambda name: name.split('-')[-1])

def get_load_dir(dataset_lvl_dir, obj_type):
    """Get load directory.

    TODO: only applicable when there is only one folder in obj_type folder.

    Args:
        dataset_lvl_dir: dataset level directory;
        obj_type: objective function type.
    Returns:
        load_dir: loading directory.
    """
    print(dataset_lvl_dir, obj_type)
    load_dir_list = glob(os.path.join(dataset_lvl_dir, obj_type, '*'))
    assert len(load_dir_list) == 1
    return load_dir_list[0]

"""Plot training vs testing curves"""
def get_evaluate_results(dataset_lvl_dir):
    """Get train and test accuracies against epochs.

    Args: 
        dataset_lvl_dir: dataset level directory
    Returns:
        x_train: a np array, epoch #s of training set;
        y_train: a np array, accuracies of training set;
        x_test: a np array, epoch #s of test set;
        x_test: a np array, accuracies of test set;
        title: a str, value = {model} - {dataset}
    """
    
    """Get title for the subplot"""
    title = '-'.join(dataset_lvl_dir.split('/')[-2:])

    """Extract data"""
    evaluate_dir = os.path.join(dataset_lvl_dir, 'evaluate')

    def extract_data(split, evaluate_dir):
        """Extract data from txt file.

        Args:
            split: a str, 'train' or 'test';
            evaluate_dir: a str, evaluate directory to load history files.
        Returns:
            x_arr: a np array, epochs of history;
            y_arr: a np array, epochs of accuracies;
        """
        # join file name 
        fpath = os.path.join(evaluate_dir, '%s_history.txt' % split)
        # load data
        raw_data = np.loadtxt(fpath, delimiter=',')
        # get accuracy array 
        y_arr = raw_data[:, 1]
        # since the first column of history matrix is an 
        # array of steps and we saved checkpoints every 10 epochs,
        # we mannually create x_arr using an array with the same 
        # length as y_arr, then times 10.
        x_arr = np.arange(1, y_arr.shape[0] + 1) * 10
        return x_arr, y_arr
    
    # extract train and test histories
    x_train, y_train = extract_data('train', evaluate_dir)
    x_test, y_test = extract_data('test', evaluate_dir)

    return (x_train, y_train), (x_test, y_test), title

def plot_train_vs_test(data_dir):
    """Plot train vs test curves for each model trained on different datasets.

    # of cols = # of models
    # of rows = # of datasets
    # of subplots = # of cols * # of rows
    
    Args:
        data_dir: data directory to folder 'final'.
    """

    """Define model level directories and model-dataset level directories"""
    model_lvl_dirs = get_model_lvl_dirs(data_dir, '*')
    model_dataset_lvl_dirs = [get_dataset_lvl_dirs(model_lvl_dir, '*')
                              for model_lvl_dir in model_lvl_dirs]
    
    """Define canvas"""
    # get #s of cols and rows
    ncols, nrows = np.array(model_dataset_lvl_dirs).shape
    # define subplots
    _, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(nrows*3, ncols*6))
    # reshape axes
    axes = np.reshape(axes, (nrows, ncols))
    
    """Plot curves"""
    for r in range(nrows):
        for c in range(ncols):
            # define an ax obj for convenient accessability
            ax = axes[r, c]

            # load train and test history accuracies against epochs
            (x_train, y_train), (x_test, y_test), title = get_evaluate_results(model_dataset_lvl_dirs[c][r])

            # plot curves
            legends = ['train', 'test']
            ax.plot(x_train, y_train)
            ax.plot(x_test, y_test)
            ax.legend(legends, loc='upper left')
            ax.set(xlabel='epoch', ylabel='accuracy', title=title)
    plt.tight_layout()
    plt.show()

"""Norm aspect"""
def compare_Ori_vs_Tar(model_dataset_lvl_dir, obj_type, instance_num, cap_idx, 
                       diffOris_vs_sameTar=True, selected_iter_ns=AVAILABLE_ITER_NS):
    """Given the instance number = {instance_num},

            if {diffOris_vs_sameTar} == True:
                 target image class = {cap_idx}
            else:
                 original image class = {cap_idx}

                 selected iteration numbers = {selected_iter_ns},
    to compare the results on different target classes given the same original
    image class.

    available iteration numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   10, 20, 40, 60, 80,
                                   100, 200, 400, 600, 800, 1000]
    
    # of rows = # of classes
    # of cols = # of iterations

    Args:
        model_dataset_lvl_dir: model-dataset level directory;
        obj_type: objective function type;
        instance_num: instance number of the example;
        cap_idx: original class index;
        selected_iter_ns: selected iteration numbers to visualize.
    """

    """Get dataset type"""
    dataset = os.path.basename(model_dataset_lvl_dir)

    """Get model type"""
    model_type = model_dataset_lvl_dir.split('/')[-2]

    """Process selected iteration numbers"""
    # filter out those indices that are not available.
    selected_iter_ns = [iter_n for iter_n in selected_iter_ns
                        if iter_n in AVAILABLE_ITER_NS]
    # add 0 iteration at the beginning
    selected_iter_ns = [0] + selected_iter_ns

    """Define canvas"""
    nrows, ncols = 10, len(selected_iter_ns)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.2, nrows*1.2))
    
    # custom legends
    legends = [Patch(facecolor='white', label='A. Original image prediction is correct;'),
               Patch(facecolor='white', label='B. Original image prediction matches Processed image prediction;'),
               Patch(facecolor='black', label='A && B;'),
               Patch(facecolor='blue', label='!A && B;'),
               Patch(facecolor='red', label='A && !B;'),
               Patch(facecolor='magenta', label='!A && !B;')]
    fig.legend(handles=legends, loc='upper center', bbox_to_anchor=(0.5, 1.1))

    if diffOris_vs_sameTar:
        fig.text(0., 0.5, 'Original Class', va='center', rotation='vertical')
    else:
        fig.text(0., 0.5, 'Target Class', va='center', rotation='vertical')
    fig.text(0.5, 1.0, 'Iteration', ha='center')
    axes = np.reshape(axes, (nrows, ncols))

    """Get data paths"""
    # get load directory
    load_dir = get_load_dir(model_dataset_lvl_dir, obj_type)

    """Plot visualizations"""
    for i in range(nrows): # classes
        for j, iter_n in enumerate(selected_iter_ns): # iterations
            if diffOris_vs_sameTar:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, i, cap_idx)))
            else:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, cap_idx, i)))
            
            # define ax for convenient access
            ax = axes[i, j]

            """Plot images"""
            # find the index of target iteration number in loaded data
            tar_idx = data['iters'].tolist().index(iter_n)
            # load images, original (class, probability), predicted (class, probability)
            img_raw = np.clip(np.squeeze(data['images'][tar_idx], axis=0), 0., 1.) # (1, 24, 24) or (3, 24, 24)
            ori_cl, ori_p = data['ori'][tar_idx].tolist()
            pred_cl, pred_p = data['pred'][tar_idx].tolist()

            # process image
            img = np.transpose(img_raw, [1, 2, 0])
            img = np.squeeze(img)

            """Adding text"""
            # disable x, y axes
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if diffOris_vs_sameTar:
                if i != ori_cl: # if the original label != the current capsule --> blue
                    if pred_cl != ori_cl: # if the predicted label != original class --> blue + red
                        ax.xaxis.label.set_color('magenta')
                    else: # --> blue
                        ax.xaxis.label.set_color('blue') 
                else: # --> red
                    if pred_cl != ori_cl:
                        ax.xaxis.label.set_color('red')
            else:
                if cap_idx != ori_cl: # --> blue
                    if pred_cl != ori_cl: # --> blue + red
                        ax.xaxis.label.set_color('magenta')
                    else: # --> blue
                        ax.xaxis.label.set_color('blue')
                else:
                    if pred_cl != ori_cl:
                        ax.xaxis.label.set_color('red')
            
            if i == 0:
                ax.set_title(iter_n)
            if j == 0:
                ax.set_ylabel(categories[dataset][i])
        
            xlabel = '(%d:' % int(ori_cl) + ('%.1f,\n' % ori_p) + '%d:' % int(pred_cl) + ('%.1f)' % pred_p)
            ax.set_xlabel(xlabel)

            # show image
            if len(img.shape) == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')

    plt.tight_layout()
    plt.show()

"""Direction aspect"""
def compare_mostActiveCap_vs_diffDims(model_dataset_lvl_dir, obj_type, instance_num, cap_idx,
                                      selected_iter_ns=AVAILABLE_ITER_NS):
    """Given the instance number = {instance_num},
                 selected capsule index = {cap_idx},
                 selected iteration numbers = {selected_iter_ns},
    to compare the results on different target dimensions given the same original
    image whose class = selected capsule index.

    available iteration numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   10, 20, 40, 60, 80,
                                   100, 200, 400, 600, 800, 1000]
    
    # of rows = # of classes
    # of cols = # of iterations

    Args:
        model_dataset_lvl_dir: model-dataset level directory;
        obj_type: objective function type;
        instance_num: instance number of the example;
        cap_idx: selected capsule index;
        selected_iter_ns: selected iteration numbers to visualize.
    """
    
    """Get dataset type"""
    dataset = os.path.basename(model_dataset_lvl_dir)

    """Process selected iteration numbers"""
    # filter out those indices that are not available.
    selected_iter_ns = [iter_n for iter_n in selected_iter_ns
                        if iter_n in AVAILABLE_ITER_NS]
    # add 0 iteration at the beginning
    selected_iter_ns = [0] + selected_iter_ns

    """Define canvas"""
    nrows, ncols = 16, len(selected_iter_ns)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.2, nrows*1.2))

    legends = [Patch(facecolor='white', label='A. Original image prediction is correct;'),
               Patch(facecolor='white', label='B. Original image prediction matches Processed image prediction;'),
               Patch(facecolor='black', label='A && B;'),
               Patch(facecolor='blue', label='!A && B;'),
               Patch(facecolor='red', label='A && !B;'),
               Patch(facecolor='magenta', label='!A && !B;')]
    fig.legend(handles=legends, loc='upper right', bbox_to_anchor=(0.5, 1.15))


    fig.text(0.5, 1., 'Iteration', ha='center')
    fig.text(0., 0.5, 'Target Dimension', va='center', rotation='vertical')
    axes = np.reshape(axes, (nrows, ncols))

    """Get data paths"""
    # get load directory
    load_dir = get_load_dir(model_dataset_lvl_dir, obj_type)

    """Plot visualizations"""
    for i in range(nrows):
        for j, iter_n in enumerate(selected_iter_ns):
            data = np.load(os.path.join(
                load_dir, 'instance_{}-cap_{}-dim_{}.npz'.format(instance_num, cap_idx, i)))
            
            # define ax for convenient access
            ax = axes[i, j]

            """Plot images"""
            # find the index of target iteration number in loaded data
            tar_idx = data['iters'].tolist().index(iter_n)
            # load images, original (class, probability), predicted (class, probability)
            img_raw = np.clip(np.squeeze(data['images'][tar_idx], axis=0), 0., 1.) # (1, 24, 24) or (3, 24, 24)
            ori_cl, ori_p = data['ori'][tar_idx].tolist()
            pred_cl, pred_p = data['pred'][tar_idx].tolist()

            # process image
            img = np.transpose(img_raw, [1, 2, 0])
            img = np.squeeze(img)

            """Adding text"""
            # disable x, y axes
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if cap_idx != ori_cl: # if the original label != the current capsule --> blue
                if pred_cl != ori_cl: # if the predicted label != original class --> blue + red
                    ax.xaxis.label.set_color('magenta')
                else: # --> blue
                    ax.xaxis.label.set_color('blue') 
            else: 
                if pred_cl != ori_cl: # --> red
                    ax.xaxis.label.set_color('red')

            if i == 0:
                ax.set_title(iter_n)
            if j == 0:
                ax.set_ylabel(i)
            xlabel = '(%d:' % int(ori_cl) + ('%.2f,' % ori_p)[1:] + '%d:' % int(pred_cl) + ('%.2f)' % pred_p)[1:]
            ax.set_xlabel(xlabel)

            # show image
            if len(img.shape) == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')

    plt.tight_layout()
    plt.show()