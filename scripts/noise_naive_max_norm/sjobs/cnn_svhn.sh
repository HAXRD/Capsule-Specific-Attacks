#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=cnn_svhn-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --mode=noise_naive_max_norm --data_dir=/home/xuc/DeepDream/data/svhn/ --dataset=svhn --max_epochs=10 --iter_n=100 --summary_dir=/home/xuc/projects/def-sageev/xuc/nov_13/cnn_svhn_ep1000_tbs_200 --total_batch_size=1 --model=cnn
