import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def d_sigmoid(z):
    y = sigmoid(z)
    return np.multiply(y, 1. - y)


def sigmoid_backward(dA, z):
    return np.multiply(dA, d_sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return z >= 0


def relu_backward(dA, z):
    return np.multiply(dA, d_relu(z))


def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


def d_tanh(z):
    return 1. - np.power(tanh(z), 2)


def tanh_bacward(dA, z):
    return np.multiply(dA, d_tanh(z))



