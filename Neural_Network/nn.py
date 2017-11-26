# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


class Layer:
    def forward(self):
        pass

    def backward(self):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, input_data, num_output, activate_function_type):
        self.input = np.mat(input_data)
        self.numOutput = num_output
        self.type = activate_function_type
        self.weights = np.mat(np.zeros((self.input.shape[0], self.numOutput)))
        self.bias = np.mat(np.zeros((self.numOutput, 1)))

    def forward(self):
        if self.type == 'sigmoid':
            output = sigmoid(self.weights.transpose() * self.input + self.bias)
        else:
            output = sigmoid(self.weights.transpose() * self.input + self.bias)
        return output

    def backward(self):
        pass


class ConvolutionLayer(Layer):
    def __init__(self, input_data, size, padding, stride):
        self.input = input_data
        self.size = size
        self.padding = padding
        self.stride = stride

    def forward(self):
        pass

    def backward(self):
        pass


class PoolingLayer(Layer):
    def __init__(self, input_data, size, stride, pooling_type):
        self.input = input_data
        self.size = size
        self.stride = stride
        self.type = pooling_type

    def forward(self):
        pass

    def backward(self):
        pass


class ReluLayer(Layer):
    def __init__(self, input_data):
        self.input = input_data

    def forward(self):
        pass

    def backward(self):
        pass


class BatchNormLayer(Layer):
    def __init__(self, input_data, eps):
        self.input = input_data
        self.eps = eps

    def forward(self):
        pass

    def backward(self):
        pass


class Net:
    def __init__(self, ):
        pass
