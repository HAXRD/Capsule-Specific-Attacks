#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-05:00      # time (DD-HH:MM)
#SBATCH --output=noise_naive_max_norm/outs/fmnist-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --total_batch_size=1 --mode=noise_naive_max_norm --data_dir=/home/xuc/DeepDream/data/fashion_mnist/ --dataset=fashion_mnist --summary_dir=/home/xuc/projects/def-sageev/xuc/final/caps_full/fmnist 

# debug
# python ~/DeepDream/experiment.py --total_batch_size=1 --mode=noise_naive_max_norm --data_dir=/home/xuc/DeepDream/data/fashion_mnist/ --dataset=fashion_mnist --iter_n=100 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/caps_full/fmnist 

