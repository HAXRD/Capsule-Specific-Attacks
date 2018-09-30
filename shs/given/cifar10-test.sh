#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4000M         # memory per node
#SBATCH --time=0-00:30      # time (DD-HH:MM)
#SBATCH --output=~/storage/DeepDream/given/cifar10-test-%j.out 

DATA_DIR=~/storage/DATA_DIR
CKPT_DIR=~/storage/CKPT_DIR

module load cuda cudnn python/2.7.14
source ~/tfp2714/bin/activate

python experiment.py --data_dir=$DATA_DIR --train=false --dataset=cifar10 \
--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false \
--summary_dir=/tmp/ --checkpoint=$CKPT_DIR/cifar/cifar{}/model.ckpt-600000 \
--num_trials=7