"""
Appendix material
Plot 2 different tight layout adv. results of dim for each dataset (MNIST and CIFAR10)
"""

import os 
import numpy as np
from glob import glob

import scipy.misc

import utils

EDGE_SIZE = 1
IMAGE_SIZE = 28 
BLOCK_SIZE = IMAGE_SIZE + 2*EDGE_SIZE

def compare_targetCap_vs_diffDims(model_dataset_lvl_dir, obj_type, instance_num, cap_idx, 
                                  selected_iter_ns=utils.AVAILABLE_ITER_NS):
    dataset = os.path.basename(model_dataset_lvl_dir)
    model = model_dataset_lvl_dir.split('/')[-2]

    # Filter selected iterations
    selected_iter_ns = [0] + [iter_n for iter_n in selected_iter_ns
                              if iter_n in utils.AVAILABLE_ITER_NS]

    canvas = np.ones((BLOCK_SIZE*16, BLOCK_SIZE*len(selected_iter_ns), 3)) 
    nrows, ncols = 16, len(selected_iter_ns)

    load_dir = utils.get_load_dir(model_dataset_lvl_dir, obj_type)

    for i in range(nrows):
        for j, iter_n in enumerate(selected_iter_ns):
            data = np.load(os.path.join(
                load_dir, 'instance_{}-cap_{}-dim_{}.npz'.format(instance_num, cap_idx, i)))
            
            tar_idx = data['iters'].tolist().index(iter_n)
            img_raw = np.clip(np.squeeze(data['images'][tar_idx], axis=0), 0., 1.)
            img = np.transpose(img_raw, [1, 2, 0])

            curr_pred = data['pred'][tar_idx]
            if tar_idx == 0:
                ori_pred_cl = np.argmax(curr_pred)
            curr_pred_cl = np.argmax(curr_pred)

            if cap_idx != ori_pred_cl:
                if curr_pred_cl != ori_pred_cl:
                    # blue + red 
                    canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                else:
                    # blue 
                    canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 0] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                    canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            else:
                if curr_pred_cl != ori_pred_cl:
                    # red
                    canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                    canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 2] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            canvas[i*BLOCK_SIZE+EDGE_SIZE:(i+1)*BLOCK_SIZE-EDGE_SIZE, j*BLOCK_SIZE+EDGE_SIZE:(j+1)*BLOCK_SIZE-EDGE_SIZE] = img

    ins_cap_dir = "ins{}_cap{}".format(instance_num, cap_idx)
    if not os.path.exists(ins_cap_dir):
        os.makedirs(ins_cap_dir)

    out_fname = "{}_{}_{}.png".format(dataset, model, obj_type)
    scipy.misc.imsave(os.path.join(ins_cap_dir, out_fname), canvas)

if __name__ == '__main__':
    data_dir = '/Users/xu/Storage/vis'
    instance_num = 3
    cap_idx = 7
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'mnist')[0]
    compare_targetCap_vs_diffDims(model_dataset_lvl_dir, 
                                  obj_type='naive_max_caps_dim', 
                                  instance_num=instance_num, 
                                  cap_idx=cap_idx)
    compare_targetCap_vs_diffDims(model_dataset_lvl_dir, 
                                  obj_type='max_caps_dim_diff', 
                                  instance_num=instance_num, 
                                  cap_idx=cap_idx)
    ######
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'cifar10')[0]
    compare_targetCap_vs_diffDims(model_dataset_lvl_dir, 
                                  obj_type='naive_max_caps_dim', 
                                  instance_num=instance_num, 
                                  cap_idx=cap_idx)
    compare_targetCap_vs_diffDims(model_dataset_lvl_dir, 
                                  obj_type='max_caps_dim_diff', 
                                  instance_num=instance_num, 
                                  cap_idx=cap_idx)
    
