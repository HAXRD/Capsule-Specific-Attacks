"""
Experiment material
Plot a folder of images, each contains a row of adv. images; and corresponding distributions (MNIST and CIFAR10)
"""

import os 
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 

import utils

def compare_Ori_vs_Tar(model_dataset_lvl_dir, obj_type, instance_num, cap_idx, 
                       diffOris_vs_sameTar=True, 
                       selected_iter_ns=utils.AVAILABLE_ITER_NS):
    
    dataset = os.path.basename(model_dataset_lvl_dir)
    model = model_dataset_lvl_dir.split('/')[-2]

    selected_iter_ns = [0] + [iter_n for iter_n in selected_iter_ns
                              if iter_n in utils.AVAILABLE_ITER_NS]
    
    nrows, ncols = 10, len(selected_iter_ns)

    load_dir = utils.get_load_dir(model_dataset_lvl_dir, obj_type)

    save_dir = "{}_{}_{}_ins{}_cap{}".format(dataset, model, obj_type, instance_num, cap_idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(nrows): # classes
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*1.4, 1.4))
        axes = np.reshape(axes, (1, ncols))
        for j, iter_n in enumerate(selected_iter_ns):
            if diffOris_vs_sameTar:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, i, cap_idx)))
            else:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, cap_idx, i)))
            
            ax = axes[0, j]

            tar_idx = data['iters'].tolist().index(iter_n)
            img_raw = np.clip(np.squeeze(data['images'][tar_idx], axis=0), 0., 1.)
            img_raw = np.transpose(img_raw, [1, 2, 0])
            img = np.squeeze(img_raw)

            curr_pred = data['pred'][tar_idx]
            if tar_idx == 0:
                ori_pred_cl = np.argmax(curr_pred)
            curr_pred_cl = np.argmax(curr_pred)

            ori_pred_prob = curr_pred[ori_pred_cl]
            curr_pred_prob = curr_pred[curr_pred_cl]

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if diffOris_vs_sameTar:
                if i != ori_pred_cl:
                    if curr_pred != ori_pred_cl:
                        ax.xaxis.label.set_color('magenta')
                    else:
                        ax.xaxis.label.set_color('blue')
                else:
                    if curr_pred_cl != ori_pred_cl:
                        ax.xaxis.label.set_color('red')
            else:
                if cap_idx != ori_pred_cl:
                    if curr_pred_cl != ori_pred_cl:
                        ax.xaxis.label.set_color('magenta')
                    else:
                        ax.xaxis.label.set_color('blue')
                else:
                    if curr_pred_cl != ori_pred_cl:
                        ax.xaxis.label.set_color('red')
            
            ax.set_title(iter_n)
            if j == 0:
                ax.set_ylabel(utils.categories[dataset][i])
            
            xlabel = '%d~' % int(ori_pred_cl) + '%.4f\n' % ori_pred_prob + \
                     '%d~' % int(curr_pred_cl) + '%.4f' % curr_pred_prob
            ax.set_xlabel(xlabel)

            if len(img.shape) == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, str(i)))

def compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, obj_type, instance_num, cap_idx, 
                                    diffOris_vs_sameTar=True, 
                                    selected_iter_ns=utils.AVAILABLE_ITER_NS):
    dataset = os.path.basename(model_dataset_lvl_dir)
    model = model_dataset_lvl_dir.split('/')[-2]

    selected_iter_ns = [0] + [iter_n for iter_n in selected_iter_ns 
                              if iter_n in utils.AVAILABLE_ITER_NS]
    nrows, ncols = 10, len(selected_iter_ns)

    barwidth = .6
    opacity = .5

    load_dir = utils.get_load_dir(model_dataset_lvl_dir, obj_type)

    save_dir = "{}_{}_{}_ins{}_cap{}".format(dataset, model, obj_type, instance_num, cap_idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(nrows):
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*2.0, 2.0))
        axes = np.reshape(axes, (1, ncols))
        
        for j, iter_n in enumerate(selected_iter_ns):
            if diffOris_vs_sameTar:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, i, cap_idx)))
            else:
                data = np.load(os.path.join(
                    load_dir, 'instance_{}-lbl0_{}-lbl1_{}.npz'.format(instance_num, cap_idx, i)))
            
            ax = axes[0, j]
            tar_idx = data['iters'].tolist().index(iter_n)
            if model == 'cnn':
                ax.set_ylim([-150.0, 50.0])
                
            curr_pred = data['pred'][tar_idx]
            if tar_idx == 0:
                ori_pred_cl = np.argmax(curr_pred)
            curr_pred_cl = np.argmax(curr_pred)

            ori_mean = np.array(curr_pred[ori_pred_cl])
            ori_index = np.array(ori_pred_cl)
            ax.bar(ori_index, ori_mean, barwidth, alpha=opacity, color='g')

            curr_mean = np.array(curr_pred[curr_pred_cl])
            curr_index = np.array(curr_pred_cl)

            ax.bar(curr_index, curr_mean, barwidth, alpha=opacity, color='r')

            depletion = np.stack((ori_index, curr_index), axis=0)
            rest_means = np.delete(curr_pred, depletion)
            rest_indices = np.delete(np.arange(10), depletion)

            ax.bar(rest_indices, rest_means, barwidth, alpha=opacity, color='b')

            ax.set_title(iter_n)
            if j == 0:
                ax.set_ylabel(utils.categories[dataset][i])
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, str(i)+'_distr'))
        plt.close()


if __name__ == '__main__':
    selected_iter_ns = [1, 2, 4,
                        10, 20, 40, 80, 100]
    data_dir = '/Users/xu/Storage/vis'
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'mnist')[0]
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)

    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cnn')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'mnist')[0]
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)

    ####
    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cap')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'cifar10')[0]
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)

    model_lvl_dir = utils.get_model_lvl_dirs(data_dir, 'cnn')[0]
    model_dataset_lvl_dir = utils.get_dataset_lvl_dirs(model_lvl_dir, 'cifar10')[0]
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='naive_max_norm', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    
    compare_Ori_vs_Tar(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)
    compare_Ori_vs_Tar_Distribution(model_dataset_lvl_dir, 
                       obj_type='max_norm_diff', 
                       instance_num=3, 
                       cap_idx=5, 
                       diffOris_vs_sameTar=True,
                       selected_iter_ns=selected_iter_ns)