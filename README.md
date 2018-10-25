# DeepDream 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Introduction
This repository demonstrates the comparisons on DeepDream between several different architectures. Due to the fact that different trained baseline models have different validation/test accuracy, we tried to keep those accuraies as close as possible. The target test models are listed as below:

| Mode            | Dataset  | CNN      | Cap      |
|-----------------|----------|----------|----------|
| train           | cifar10  | done     | done     |
|                 | mnist    | done     | done     |
| test            | cifar10  | done     | done     |
|                 | mnist    | done     | done     |
| naive           | cifar10  | done     | done     |
|                 | mnist    | done     | done     |
| multiscale      | cifar10  | done     | done     |
|                 | mnist    | done     | done     |
| naive_dream     | mnist    | done     | done     |

## Requirements

* Tensorflow-GPU
* 1 GPU

## Download dataset and ckpts

* Download and extract cifar10 binary version to $DATA_DIR/
from https://www.cs.toronto.edu/~kriz/cifar.html
* Download mnist to $DATA_DIR/mnist with 
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

* Download and extract capsule ckpts trained on mnist dataset to $SUMMARY_DIR/
from https://storage.googleapis.com/ckpts/cap_mnist.zip
* Download and extract cnn ckpts trained on mnist dataset to $SUMMARY_DIR/
from https://storage.googleapis.com/ckpts/cnn_mnist.zip

## Inspect model parameters

```
python experiment -h
```

## Train the model
### CNN on cifar10
Note: the batch_size is totally depending on your GPU capcity.
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/cifar-10-batches-bin/ --dataset=cifar10 --max_epochs=500 --mode=train --model=cnn --save_epochs=5 --summary_dir=$SUMMARY_DIR/cnn_cifar10
```

### CNN on mnist
Note: the batch_size is totally depending on your GPU capcity.
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/mnist --dataset=mnist --max_epochs=500 --mode=train --model=cnn --save_epochs=5 --summary_dir=$SUMMARY_DIR/cnn_mnist
```

### Cap on cifar10
Note: the batch_size is totally depending on your GPU capcity.
```
python experiment.py --batch_size=50 --data_dir=$DATA_DIR/cifar-10-batches-bin/ --dataset=cifar10 --hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false --max_epochs=500 --mode=train --model=cap --save_epochs=5 --summary_dir=$SUMMARY_DIR/cap_cifar10
```

### Cap on mnist
Note: the batch_size is totally depending on your GPU capcity.
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/mnist/ --dataset=mnist --max_epochs=500 --mode=train --model=cap --save_epochs=5 --summary_dir=$SUMMARY_DIR/cap_mnist
```

## Test the model
### CNN on cifar10
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/cifar-10-batches-bin/ --dataset=cifar10 --max_epochs=1 --mode=test --model=cnn --summary_dir=$SUMMARY_DIR/cnn_cifar10
```

### CNN on mnist
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/mnist --dataset=mnist --max_epochs=1 --mode=test --model=cnn --summary_dir=$SUMMARY_DIR/cnn_mnist
```

### Cap on cifar10
```
python experiment.py --batch_size=50 --data_dir=$DATA_DIR/cifar-10-batches-bin/ --dataset=cifar10 --max_epochs=1 --mode=test --model=cap --summary_dir=$SUMMARY_DIR/cap_cifar10
```

### Cap on mnist
```
python experiment.py --batch_size=100 --data_dir=$DATA_DIR/mnist/ --dataset=mnist --max_epochs=1 --mode=test --model=cap --summary_dir=$SUMMARY_DIR/cap_mnist
```

## Naive or Multiscale feature visualizations
### CNN on cifar10
```
python experiment.py --batch_size=1 --dataset=cifar10 --max_epochs=1 --mode=naive --model=cnn --summary_dir=$SUMMARY_DIR/cnn_cifar10
```

### CNN on mnist
```
python experiment.py --batch_size=1 --dataset=mnist --max_epochs=1 --mode=naive --model=cnn --summary_dir=$SUMMARY_DIR/cnn_mnist
```

### Cap on cifar10
```
python experiment.py --batch_size=1 --dataset=cifar10 --max_epochs=1 --mode=naive --model=cap --summary_dir=$SUMMARY_DIR/cap_cifar10
```

### Cap on mnist
```
python experiment.py --batch_size=1/ --dataset=mnist --max_epochs=1 --mode=naive --model=cap --summary_dir=$SUMMARY_DIR/cap_mnist
```

## Naive Dream
### CNN on cifar10
```
python experiment.py --batch_size=1 --dataset=cifar10 --max_epochs=1 --mode=dream --model=cnn --summary_dir=$SUMMARY_DIR/cnn_cifar10
```

### CNN on mnist
```
python experiment.py --batch_size=1 --dataset=mnist --max_epochs=1 --mode=dream --model=cnn --summary_dir=$SUMMARY_DIR/cnn_mnist
```

### Cap on cifar10
```
python experiment.py --batch_size=1 --dataset=cifar10 --max_epochs=1 --mode=dream --model=cap --summary_dir=$SUMMARY_DIR/cap_cifar10
```

### Cap on mnist
```
python experiment.py --batch_size=1/ --dataset=mnist --max_epochs=1 --mode=dream --model=cap --summary_dir=$SUMMARY_DIR/cap_mnist
```


## References
### Tensorflow:
* [CNN for MNIST](http://parneetk.github.io/blog/cnn-mnist/)
* [Capsule for MNIST](https://github.com/XifengGuo/CapsNet-Keras)
* [DeepDream on GoogLeNet && VGG16](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)