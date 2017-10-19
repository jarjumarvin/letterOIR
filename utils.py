#!/usr/bin/env python3.6
"""downloads and prepares MNIST data for training and validation."""
import os
import gzip
import _pickle
import wget
import numpy as np
import random
from matplotlib import pyplot as plt


def load_mnist():
    """Download mnist data and return train, test and validation data."""
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                      out='data')

    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'),
                          'rb')
    training_data, validation_data, test_data = _pickle.load(data_file,
                                                             encoding='latin1')
    data_file.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = validation_data[1]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data


def vectorized_result(y):
    """Return Vectorized Result."""
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def plot_prediction(x, prediction):
    plt.imshow(x.reshape((28, 28)), cmap='gray')
    plt.title('Network\'s Prediction: {0}'.format(prediction))
    plt.axis('off')
    plt.show()


def get_random_image(data):
    list_x = [list(t) for t in zip(*data)]
    return list_x[0][random.randint(0, len(data))]
