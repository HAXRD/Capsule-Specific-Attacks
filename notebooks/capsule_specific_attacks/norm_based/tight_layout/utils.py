"""
Utility file functions 
"""

import os 
from glob import glob

AVAILABLE_ITER_NS = [1, 2, 3, 4, 5,
                     10, 20, 40, 60, 80, 100]

def get_model_lvl_dirs(data_dir, model_pattern='*'):
    """
    Get model level directories.
    
    :param data_dir: directory to data
    :param model_pattern: patterns to match the model, '*' find all the models
    :return model_lvl_dirs: a list, model level directories
    """
    return sorted(glob(os.path.join(data_dir, model_pattern)))

def get_dataset_lvl_dirs(model_lvl_dir, dataset_pattern='*'):
    """
    Get dataset level directories.
    
    :param model_lvl_dir: model level directory
    :param dataset_pattern: patterns to match the dataset, '*' find all the datasets
    :return dataset_dirs: a list, dataset level directories
    """
    return sorted(glob(
        os.path.join(model_lvl_dir, dataset_pattern)), 
        key=lambda name: name.split('-')[-1])

def get_load_dir(dataset_lvl_dir, obj_type):
    """
    Get load directory.

    :param dataset_lvl_dir: dataset level directory
    :param obj_type: objective function type
    :returns load_dir: loading directory
    """
    load_dir_list = glob(os.path.join(dataset_lvl_dir, obj_type, '*'))
    return load_dir_list[0]

