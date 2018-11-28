#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-05:00      # time (DD-HH:MM)
#SBATCH --output=max_norm_diff/outs/mnist-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --total_batch_size=1 --mode=max_norm_diff --data_dir=/home/xuc/DeepDream/data/mnist/ --dataset=mnist --max_epochs=10 --iter_n=1000 --summary_dir=/home/xuc/projects/def-sageev/xuc/final_24_0.1/cnn/mnist --model=cnn --step=0.1 --image_size=24

# debug
# python ~/DeepDream/experiment.py --total_batch_size=1 --mode=max_norm_diff --data_dir=/home/xuc/DeepDream/data/mnist/ --dataset=mnist --max_epochs=10 --iter_n=100 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/cnn/mnist --model=cnn --step=0.1 --image_size=24

