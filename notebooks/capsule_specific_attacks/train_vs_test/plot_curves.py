import os 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

def get_evaluate_results(file_list):
    title = file_list[0].split('/')[-3].upper()
    logger.debug(title)
    
    def extract_data(path):
        raw_data = np.loadtxt(path, delimiter=',')
        y_arr = raw_data[:, 1]
        x_arr = np.arange(1, y_arr.shape[0] + 1) * 10
        logger.debug("{}{}".format(x_arr.shape, y_arr.shape))
        return (x_arr, y_arr)
    
    result_pairs = [extract_data(file_path) for file_path in file_list]
    return result_pairs, title

def plot_train_vs_test(src_dir):
    MNIST_CNN_TRAIN_PATH = glob(os.path.join(src_dir, 'cnn', 'mnist', '*/train_history*'))
    MNIST_CAP_TRAIN_PATH = glob(os.path.join(src_dir, 'cap', 'mnist', '*/train_history*'))
    MNIST_CNN_TEST_PATH = glob(os.path.join(src_dir, 'cnn', 'mnist', '*/test_history*'))
    MNIST_CAP_TEST_PATH = glob(os.path.join(src_dir, 'cap', 'mnist', '*/test_history*'))
    MNIST_FILE_LIST = MNIST_CNN_TRAIN_PATH + MNIST_CAP_TRAIN_PATH + MNIST_CNN_TEST_PATH + MNIST_CAP_TEST_PATH

    CIFAR10_CNN_TRAIN_PATH = glob(os.path.join(src_dir, 'cnn', 'cifar10', '*/train_history*'))
    CIFAR10_CAP_TRAIN_PATH = glob(os.path.join(src_dir, 'cap', 'cifar10', '*/train_history*'))
    CIFAR10_CNN_TEST_PATH = glob(os.path.join(src_dir, 'cnn', 'cifar10', '*/test_history*'))
    CIFAR10_CAP_TEST_PATH = glob(os.path.join(src_dir, 'cap', 'cifar10', '*/test_history*'))
    CIFAR10_FILE_LIST = CIFAR10_CNN_TRAIN_PATH + CIFAR10_CAP_TRAIN_PATH + CIFAR10_CNN_TEST_PATH + CIFAR10_CAP_TEST_PATH
    
    FILE_LISTS = [MNIST_FILE_LIST, CIFAR10_FILE_LIST]
    
    # Plot 
    nrows, ncols = [1, 2]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4))
    axes = np.reshape(axes, (nrows, ncols))
    
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            result_pairs, title = get_evaluate_results(FILE_LISTS[r*2 + c])
            legends = ['train (cnn)', 'train (caps)', 'test (cnn)', 'test (caps)']
            for idx, pair in enumerate(result_pairs):
                if idx < 2:
                    ax.plot(pair[0], pair[1])
                else: 
                    ax.plot(pair[0], pair[1], '--')
            ax.legend(legends, loc='lower right')
            ax.set(xlabel='epoch', ylabel='accuracy', title=title)
    plt.tight_layout()
    # plt.show()
    fig.savefig('curve.png')

if __name__ == '__main__':
    data_dir = '/Users/xu/Storage/vis'
    plot_train_vs_test(data_dir)