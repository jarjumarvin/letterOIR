#!/usr/bin/env python3.6
# Includes multiple activation functions and their derivatives
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - tanh(z) * tanh(z)


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return float(z > 0)


def get_prime_function(function):
    if function == sigmoid:
        return sigmoid_prime
    elif function == softmax:
        return softmax_prime
    elif function == tanh:
        return tanh_prime
    elif function == relu:
        return relu_prime
