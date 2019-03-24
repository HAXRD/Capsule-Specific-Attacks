#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-05:00      # time (DD-HH:MM)
#SBATCH --output=max_caps_dim_diff/outs/mnist-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

REPO_DIR=/home/xuc/DeepDream
TOTAL_BATCH_SIZE=1
MODEL=cap
MODE=max_caps_dim_diff
DATASET=mnist
MAX_EPOCHS=20
SUMMARY_DIR=/home/xuc/projects/def-sageev/xuc/$MODEL/$DATASET

python $REPO_DIR/experiment.py --total_batch_size=$TOTAL_BATCH_SIZE --mode=$MODE --data_dir=$REPO_DIR/data/$DATASET/ --dataset=$DATASET --summary_dir=$SUMMARY_DIR --max_epochs=$MAX_EPOCHS
