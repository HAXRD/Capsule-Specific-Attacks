#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-09:00      # time (DD-HH:MM)
#SBATCH --output=train/outs/mnist-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --total_batch_size=200 --mode=train --model=cap --data_dir=/home/xuc/DeepDream/data/mnist/ --dataset=mnist --max_epochs=1000 --summary_dir=/home/xuc/projects/def-sageev/xuc/final_24_0.1/caps_full/mnist --image_size=24

# debug
# python ~/DeepDream/experiment.py --total_batch_size=200 --mode=train --model=cap --data_dir=/home/xuc/DeepDream/data/mnist/ --dataset=mnist --max_epochs=10 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/caps_full/mnist --image_size=24

