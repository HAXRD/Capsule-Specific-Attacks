"""
Appendix material
Plot 2 different tight layout adv. results for each dataset (MNIST and CIFAR10)
"""

import os 
import numpy as np
from glob import glob
import scipy.misc

import utils

EDGE_SIZE = 1
IMAGE_SIZE = 28 
BLOCK_SIZE = IMAGE_SIZE + 2*EDGE_SIZE

def compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, obj_type, instance_num, cap_idx, 
                             diffOris_vs_sameTar=True, 
                             selected_iter_ns=utils.AVAILABLE_ITER_NS):
    
    dataset = os.path.basename(model_dataset_lvl_dir)
    model = model_dataset_lvl_dir.split('/')[-2]

    # Filter selected iterations
    selected_iter_ns = [0] + [iter_n for iter_n in selected_iter_ns
                              if iter_n in utils.AVAILABLE_ITER_NS]
    
    canvas = np.ones((BLOCK_SIZE*10, BLOCK_SIZE*len(selected_iter_ns), 3)) 
    nrows, ncols = 10, len(selected_iter_ns)

    ins_cap_dir = "ins{}_cap{}".format(instance_num, cap_idx)
    if not os.path.exists(ins_cap_dir):
        os.makedirs(ins_cap_dir)

    compare_type_dir = "Diff_Ori-Same_Tar" if diffOris_vs_sameTar else "Same_Ori-Diff_Tar"
    if not os.path.exists(os.path.join(ins_cap_dir, compare_type_dir)):
        os.makedirs(os.path.join(ins_cap_dir, compare_type_dir))
    
    load_dir = utils.get_load_dir(model_dataset_lvl_dir, obj_type)

    for i in range(nrows):
        for j, iter_n in enumerate(selected_iter_ns):
            if diffOris_vs_sameTar: 
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, i, cap_idx)))
            else:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, cap_idx, i)))
            
            tar_idx = data['iters'].tolist().index(iter_n)
            img_raw = np.clip(np.squeeze(data['images'][tar_idx], axis=0), 0., 1.)
            img = np.transpose(img_raw, [1, 2, 0])

            curr_pred = data['pred'][tar_idx]
            if tar_idx == 0:
                ori_pred_cl = np.argmax(curr_pred)
            curr_pred_cl = np.argmax(curr_pred)

            if diffOris_vs_sameTar:
                if i != ori_pred_cl:
                    if ori_pred_cl != curr_pred_cl:
                        # blue + red
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                    else:
                        # blue 
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 0] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                else:
                    if ori_pred_cl != curr_pred_cl:
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 2] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            else:
                if cap_idx != ori_pred_cl:
                    if ori_pred_cl != curr_pred_cl:
                        # blue + red
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                    else:
                        # blue 
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 0] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                else:
                    if ori_pred_cl != curr_pred_cl:
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 1] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
                        canvas[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE, 2] = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            canvas[i*BLOCK_SIZE+EDGE_SIZE:(i+1)*BLOCK_SIZE-EDGE_SIZE, j*BLOCK_SIZE+EDGE_SIZE:(j+1)*BLOCK_SIZE-EDGE_SIZE] = img

    out_fname = "{}_{}_{}.png".format(dataset, model, obj_type, instance_num, cap_idx)
    scipy.misc.imsave(os.path.join(ins_cap_dir, compare_type_dir, out_fname), canvas)

if __name__ == '__main__':
    data_dir = '/Users/xu/Storage/vis'
    instance_num = 3
    cap_idx = 7
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'mnist')[0]
    for compare_type in [True, False]:
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='naive_max_norm', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)

        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='max_norm_diff', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)

    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cnn')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'mnist')[0]
    for compare_type in [True, False]:
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='naive_max_norm', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='max_norm_diff', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)

    ######
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'cifar10')[0]
    for compare_type in [True, False]:
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='naive_max_norm', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)

        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='max_norm_diff', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)

    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cnn')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'cifar10')[0]
    for compare_type in [True, False]:
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='naive_max_norm', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)
        compare_Ori_vs_Tar_tight(model_dataset_lvl_dir, 
                                obj_type='max_norm_diff', 
                                instance_num=instance_num, 
                                cap_idx=cap_idx,
                                diffOris_vs_sameTar=compare_type)