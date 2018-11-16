#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=evaluate/outs/svhn-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --total_batch_size=200 --mode=evaluate --data_dir=/home/xuc/DeepDream/data/svhn/ --dataset=svhn --max_epochs=1 --summary_dir=/home/xuc/projects/def-sageev/xuc/final/caps_full/svhn

# debug
# python ~/DeepDream/experiment.py --total_batch_size=200 --mode=evaluate --data_dir=/home/xuc/DeepDream/data/svhn/ --dataset=svhn --max_epochs=1 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/caps_full/svhn

