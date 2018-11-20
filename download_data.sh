#!/bin/bash
wget -P data/mnist https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

wget -P data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

wget -P data/svhn http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -P data/svhn http://ufldl.stanford.edu/housenumbers/test_32x32.mat

wget -P data/ https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar -C data/ -xvzf data/cifar-10-matlab.tar.gz 