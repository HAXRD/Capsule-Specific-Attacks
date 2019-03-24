#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=evaluate/outs/cifar10-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/DeepDream/experiment.py --total_batch_size=200 --mode=evaluate --data_dir=/home/xuc/DeepDream/data/cifar-10-batches-mat/ --dataset=cifar10 --max_epochs=1 --summary_dir=/home/xuc/projects/def-sageev/xuc/final/cnn/cifar10 --model=cnn

# debug
# python ~/DeepDream/experiment.py --total_batch_size=200 --mode=evaluate --data_dir=/home/xuc/DeepDream/data/cifar-10-batches-mat/ --dataset=cifar10 --max_epochs=1 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/cnn/cifar10 --model=cnn

REPO_DIR=/home/xuc/DeepDream
TOTAL_BATCH_SIZE=200
MODEL=cnn
MODE=evaluate
DATASET=cifar10
MAX_EPOCHS=1
SUMMARY_DIR=/home/xuc/projects/def-sageev/xuc/$MODEL/$DATASET

python $REPO_DIR/experiment.py --total_batch_size=$TOTAL_BATCH_SIZE --mode=$MODE --data_dir=$REPO_DIR/data/$DATASET/ --dataset=$DATASET --max_epochs=$MAX_EPOCHS --summary_dir=$SUMMARY_DIR --model=$MODEL
