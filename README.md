# DeepDream 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for thesis (second part): [Adversarial Attacks on Capsule Neural Networks](https://github.com/XxuChen/Adversarial-Attack-on-CapsNets)

## Quick Overview
How to interpret the visualization can be found in thesis mentioned above.

More detailed results can be found in folder `scripts`. 

### Naive Max Norm (MNIST Caps)
![NMN](notebooks/capsule_specific_attacks/norm_based/tight_layout/mnist_cap_naive_max_norm_ins10_cap7.png)
### Max Norm Diff (MNIST Caps)
![MND](notebooks/capsule_specific_attacks/norm_based/tight_layout/mnist_cap_max_norm_diff_ins10_cap7.png)
### Naive Max Caps Dim (MNIST Caps)
![NMCD](notebooks/capsule_specific_attacks/dim_based/tight_layout/mnist_cap_naive_max_caps_dim_ins3_cap5.png)
### Max Caps Dim Diff (MNIST Caps)
![MCDD](notebooks/capsule_specific_attacks/dim_based/tight_layout/mnist_cap_max_caps_dim_diff_ins3_cap5.png)

## Prerequisite
- Python3
- Tensorflow
- GPU
- NumPy

## Download Datasets
```
chmod +x download_data.sh && ./download_data.sh
```

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