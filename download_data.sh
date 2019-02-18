#!/bin/bash
wget -P data/mnist https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

wget -P data/ https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar -C data/ -xvzf data/cifar-10-matlab.tar.gz 