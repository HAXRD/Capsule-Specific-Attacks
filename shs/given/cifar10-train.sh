#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=0-20:00      # time (DD-HH:MM)
#SBATCH --output=~/storage/DeepDream/given/cifar10-train-%j.out 

DATA_DIR=~/storage/DATA_DIR
CKPT_DIR=~/storage/CKPT_DIR

PROJECT_ROOT=~/HT/DeepDream/
SUM_DIR=~/storage/SUM_DIR

module load cuda cudnn python/2.7.14
source ~/tfp2714/bin/activate

python $PROJECT_ROOT/experiment.py --data_dir=$DATA_DIR --dataset=cifar10 --max_steps=600000\
--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false \
--summary_dir=$SUM_DIR/cifar10