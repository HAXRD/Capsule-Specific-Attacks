#!/bin/bash

DATA_DIR=~/storage/DATA_DIR
rm -r $DATA_DIR

wget https://storage.googleapis.com/capsule_toronto/mnist_data.tar.gz -P $DATA_DIR
tar xvzf $DATA_DIR/mnist_data.tar.gz -C $DATA_DIR
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P $DATA_DIR
tar xvzf $DATA_DIR/cifar-10-binary.tar.gz -C $DATA_DIR


CKPT_DIR=~/storage/CKPT_DIR
rm -r $CKPT_DIR

wget https://storage.googleapis.com/capsule_toronto/mnist_checkpoints.tar.gz -P $CKPT_DIR
tar xvzf $CKPT_DIR/mnist_checkpoints.tar.gz -C $CKPT_DIR
wget https://storage.googleapis.com/capsule_toronto/cifar_checkpoints.tar.gz -P $CKPT_DIR
tar xvzf $CKPT_DIR/cifar_checkpoints.tar.gz -C $CKPT_DIR
