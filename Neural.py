"""My own neural network module for building feed-forward networks trained with stochastic gradient descend."""

# Standard library
import datetime
import random

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


def shuffle_equally(arr1, arr2):
    """
    Shuffles two np.arrays equally through the first axis.
    :param arr1: np.array.
    :param arr2: np.array.
    :return: None
    """

    assert len(arr1) == len(arr2), "both arrays must be equally long; however, the first one has a length of " + \
                                   str(len(arr1)) + " while the second one has a length of " + str(len(arr2)) + ". "

    for i in range(len(arr1) - 1, 0, -1):
        j = random.randint(0, i + 1)
        arr1[i], arr1[j] = arr1[j], arr1[i]
        arr2[i], arr2[j] = arr2[j], arr2[i]


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

    def predict(self, x, long_output=False):
        """
        :param x: input for the network (numpy.array).
        :param long_output: boolean. If it's true, this would return a list with the outputs of every
        layer in order from left to right (each element is the result of Layer.feed_forward).
        :return: if long_output == False, a np.array which is the output of the network for a given input (x);
        otherwise, it's a list with the outputs of every layer in order from left to right (each element is the
        result of Layer.feed_forward).
        """

        assert x.shape == self.layers[0].input_shape, "invalid input: expected np.array of shape " + \
                                                      str(self.layers[0].input_shape) + " but an array with shape " + \
                                                      str(x.shape) + " was given. "
        outputs = []

        last_output = None
        for layer in self.layers:
            if last_output is not None:
                result = layer.feed_forward(last_output)
                last_output = result[0]
                if long_output:
                    outputs.append(result)
            else:
                result = layer.feed_forward(x)
                last_output = result[0]
                if long_output:
                    outputs.append(result)

        return outputs if long_output else last_output

    def train(self, cost_f, training_inputs, training_labels, epochs, minibatches, lr, test_inputs=None,
              test_labels=None, eval_func=np.argmax):
        """
        :param cost_f: cost function (of type Function).
        :param training_inputs: np.array with np.arrays each one with one input
        training data (without the expected output).
        :param training_labels: np.array with np.arrays each one with the expected output
        to the corresponding input in training_inputs.
        :param epochs: int, the number of epochs.
        :param minibatches: int, the number of minibatches of each epoch.
        :param lr: the learning rate for stochastic gradient descend.
        :param test_inputs: np.array with np.arrays each one with one input
        test data (without the expected output). It's not obligatory, but it's highly recommended
        (it could make the overfitting easier to detect).
        :param test_labels: np.array with np.arrays each one with the expected output
        to the corresponding input in test_inputs. It's not obligatory, but it's highly recommended
        (it could make the overfitting easier to detect).
        :param eval_func: function (type function, not Function) to use for evaluating the model on test data
        (the code passes self.predict result to this and checks if it's equal to test_labels[some_index].
        NOTE: It's not the cost function, it's just for testing the model. Default to np.argmax because my problem is
        one-hot encoded.
        :return: None
        """

        time_print("The training process is starting...\n\n")

        for epoch in range(1, epochs + 1):
            shuffle_equally(training_inputs, training_labels)
            shuffle_equally(training_inputs, training_labels)

            minibatches_inputs = np.array_split(training_inputs, minibatches)
            minibatches_labels = np.array_split(training_labels, minibatches)

            # Think of this loop as "for every minibatch..."
            for example_inputs, example_labels in zip(minibatches_inputs, minibatches_labels):
                # Do the key stuff
                pass

            if test_inputs is not None and test_labels is not None:
                correct = 0

                for test_input, test_label in zip(test_inputs, test_labels):
                    if eval_func(self.predict(test_input)) == test_label:
                        correct += 1

                self.acc = correct / len(test_labels)
                time_print(f"\nEpoch {epoch} completed: test accuracy: {self.acc}. \n")
            else:
                time_print(f"\nEpoch {epoch} completed. \n")
