#!/usr/bin/env python3.6
import os
import numpy as np
import random as r
import activationfuncs


class Network(object):

    def __init__(self, sizes=list(), learning_rate=1.0,
                 training_batch_size=16, iterations=10,
                 function=activationfuncs.sigmoid):
        """Initialize the network and it's basic parameters.

        Arguments:
        sizes - number of neurons in the network layers [input, hidden, output]
        learning_rate -- learning rate during gradient descent, default=1.0
        training_batch_size -- batch size during training, default=16
        iterations -- number of training iterations, default=10
        """
        self.sizes = sizes
        self.num_layers = len(sizes)

        # input layer has no weights, other layers have random weights
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # self.biases[0] is redundant, since input layer has no biases
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # input layer has no W and b, z=wx+b doesnt apply so _zs[0] redundant
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        #  training examples are treated as activations of the input layer
        #  so self.activations[0] = training_example_data
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.training_batch_size = training_batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.function = function
        self.function_prime = activationfuncs.get_prime_function(function)
        if not os.path.exists(os.path.join(os.curdir, 'models')):
            os.mkdir(os.path.join(os.curdir, 'models'))

    def train(self, training_data):
        """Train the network.

        perform gradient descent and back propagation
        """
        for i in range(self.iterations):
            r.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.training_batch_size] for k in
                range(0, len(training_data), self.training_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    nabla_b = [
                        nb + dnb for nb, dnb in
                        zip(nabla_b, delta_nabla_b)]
                    nabla_w = [
                        nw + dnw for nw, dnw in
                        zip(nabla_w, delta_nabla_w)]

                self.biases = [
                    b - (self.learning_rate / self.training_batch_size) * db
                    for b, db in zip(self.biases, nabla_b)
                ]
                self.weights = [
                    w - (self.learning_rate / self.training_batch_size) * dw
                    for w, dw in zip(self.weights, nabla_w)
                ]
            print('iteration {0} done'.format(i))

    def predict(self, x):
        """Return a single prediction on one piece of data."""
        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        """Forward propagate."""
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = self.function(self._zs[i])

    def _back_prop(self, x, y):
        """Backward Propagation and return adjusted delta W and B."""
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * self.function_prime(self._zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                self.function_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def validate(self, validation_data):
        """Return accuracy in %."""
        results = [(self.predict(x) == y) for x, y in validation_data]
        return np.round(((sum(result for result in results) /
                          len(validation_data)) * 100), 2)

    def load(self, filename='model.npz'):
        """Save network parameters in compressed binary.

        Default file name is model.npz
        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))
        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]
        self.training_batch_size = int(npz_members['training_batch_size'])
        self.iterations = int(npz_members['iterations'])
        self.learning_rate = float(npz_members['learning_rate'])

    def save(self, filename='model.npz'):
        """Load network parameters from compressed binary.

        Default file name is model.npz
        """
        np.savez_compressed(
            file=os.path.join('models', filename),
            weights=self.weights,
            biases=self.biases,
            training_batch_size=self.training_batch_size,
            iterations=self.iterations,
            learning_rate=self.learning_rate
        )
