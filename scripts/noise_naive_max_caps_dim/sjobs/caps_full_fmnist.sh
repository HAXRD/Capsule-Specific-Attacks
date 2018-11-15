#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=caps_full_fmnist-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --data_dir=/home/xuc/DeepDream/data/fashion_mnist/ --dataset=fashion_mnist --iter_n=100 --max_epochs=10 --mode=noise_naive_max_caps_dim --summary_dir=/home/xuc/projects/def-sageev/xuc/nov_13/caps_full_fmnist_ep1000_tbs_200 --total_batch_size=1