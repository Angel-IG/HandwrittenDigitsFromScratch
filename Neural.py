"""My own neural network module for building feed-forward networks trained with stochastic gradient descend.
"""

import numpy as np


class Function:
    """Data structure for using functions with both the normal mode and the derivative. """
    def __init__(self, evaluation, deriv):
        """
        :param evaluation: function which takes some input and returns some output.
        :param deriv: function which is the derivative of the evaluation function.
        """

        self.evaluation = evaluation
        self.deriv = deriv


class Layer:
    """One layer of a network"""
    def __init__(self, activation_func, neurons, next_neurons):
        """
        :param activation_func: object of type Function for the activation function of the layer.
        :param neurons: int to represent the number of neurons in the layer.
        :param next_neurons: int to represent the number of neurons of the next layer
        """

        self.activation_func = activation_func
        self.b = np.random.randn(next_neurons)  # Biases
        self.W = np.random.randn(neurons, next_neurons)  # Weights

    def feed_forward(self, x):
        """
        :param x: np.array with the input for the layer
        :return: tuple with two values: the output of the layer (with the activation function) and the weighted sum (without the activation function)
        """

        z = x @ self.W + self.b
        return self.activation_func.evaluation(z), z
