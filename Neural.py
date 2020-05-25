"""My own neural network module for building feed-forward networks trained with stochastic gradient descend."""

# Standard library
import datetime

# Third party
import numpy as np


def time_print(*args, **kwargs):
    """
    Prints anything with the current time before the message. Used in training.
    :param args: the same as 'print'.
    :param kwargs: the same as 'print'.
    :return: None.
    """

    print(datetime.datetime.now(), *args, **kwargs)

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
        self.input_shape = (neurons,)

    def feed_forward(self, x):
        """
        :param x: np.array with the input for the layer
        :return: tuple with two values: the output of the layer (with the activation function) and the weighted sum (without the activation function)
        """

        assert x.shape == self.input_shape, "invalid input: expected np.array of shape " + str(self.input_shape) + \
                                            " but an array with shape " + str(x.shape) + " was given. "

        z = x @ self.W + self.b
        return self.activation_func.evaluation(z), z


class Network:
    """Class to represent a feed-forward neural network"""

    def __init__(self, topology, activations):
        """
        :param topology: list with the number of neurons in each layer: the first element is the neurons in the
        first layer and the last one in the last layer.
        :param activations: list with the activation functions (type Function) of each
        hidden/output layers (len(topology) - 1 == len(activations)).
        """

        assert len(topology) - 1 == len(activations), "invalid topology or activations for the network " \
                                                      "(the length of the activations should be one less than the " \
                                                      "length of the topology)"

        self.layers = []
        for i in range(len(topology[:-1])):
            self.layers.append(Layer(activation_func=activations[i], neurons=topology[i], next_neurons=topology[i + 1]))

        self.acc = 0

    def predict(self, x):
        """
        :param x: input for the network (numpy.array).
        :return: output of the network for a given input (x).
        """

        assert x.shape == self.layers[0].input_shape, "invalid input: expected np.array of shape " + \
                                                      str(self.layers[0].input_shape) + " but an array with shape " + \
                                                      str(x.shape) + " was given. "

        last_output = None
        for layer in self.layers:
            if last_output is not None:
                last_output = layer.feed_forward(last_output)[0]
            else:
                last_output = layer.feed_forward(x)[0]

        return last_output


# Testing. This should be removed
sigmoid = Function(lambda x: 1 / (1 + np.e ** -x), lambda x: x)  # Undefined derivative for testing.
model = Network([8, 4, 2], [sigmoid, sigmoid])
time_print(model.predict(np.array([1, 5, 46, 4, 2, 1, 5, 7])))
