"""My own neural network module for building feed-forward networks trained with gradient descent."""

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
        j = random.randint(0, i)
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
        :param x: np.array with the input for the layer.
        :return: tuple with two values: the output of the layer (with the activation function) and the
        weighted sum (without the activation function).
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

        last_output = x
        for layer in self.layers:
            result = layer.feed_forward(last_output)
            last_output = result[0]
            if long_output:
                outputs.append(result)

        return outputs if long_output else last_output

    def train(self, cost_f, training_inputs, training_labels, epochs, minibatch_size, lr, test_inputs=None,
              test_labels=None, eval_func=np.argmax, print_loss=True):
        """
        :param cost_f: cost function (of type Function).
        :param training_inputs: np.array with np.arrays each one with one input
        training data (without the expected output).
        :param training_labels: np.array with np.arrays each one with the expected output
        to the corresponding input in training_inputs.
        :param epochs: int, the number of epochs.
        :param minibatch_size: int, the number of training examples of each minibatch.
        :param lr: the learning rate for mini-batch gradient descent.
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
        :param print_loss: boolean, whether the average loss should be printed in each batch or not.
        :return: None
        """

        assert len(training_labels) % minibatch_size == 0, "length of training data must " \
                                                           "be divisible by the number of minibatches. " \
                                                           "However, " + str(len(training_labels)) + \
                                                           " is not divisible by " + str(minibatch_size) + ". "

        minibatches = len(training_labels) / minibatch_size

        time_print("The training process is starting...\n\n")

        for epoch in range(1, epochs + 1):
            shuffle_equally(training_inputs, training_labels)
            # We don't have to shuffle the test data as the order is irrelevant.

            minibatches_inputs = np.array_split(training_inputs, minibatches)
            minibatches_labels = np.array_split(training_labels, minibatches)

            minibatch = 1
            # Think of this loop as "for every minibatch..."
            for example_inputs, example_labels in zip(minibatches_inputs, minibatches_labels):
                self.gradient_descent_step(example_inputs, example_labels, cost_f, lr=lr,
                                           epoch_number=epoch, minibatch_number=minibatch, print_loss=print_loss)

                minibatch += 1

            if test_inputs is not None and test_labels is not None:
                correct = 0

                for test_input, test_label in zip(test_inputs, test_labels):
                    if eval_func(self.predict(test_input)) == eval_func(test_label):
                        correct += 1

                self.acc = correct / len(test_labels)
                time_print(f"\nEpoch {epoch} completed: test accuracy: {self.acc}. \n")
            else:
                time_print(f"\nEpoch {epoch} completed. \n")

    def gradient_descent_step(self, example_inputs, example_labels, cost_f, lr, epoch_number, minibatch_number,
                              print_loss=True):
        """
        :param example_inputs: training inputs of the minibatch.
        :param example_labels: training labels of the minibatch.
        :param cost_f: Function, cost function of the network.
        :param lr: learning rate.
        :param epoch_number: current epoch. Used just for prining.
        :param minibatch_number: current minibatch. Used just for printing.
        :param print_loss: whether the average loss should be printed or not.
        :return: None
        """

        bias_derivatives = list(np.zeros((len(self.layers),)))
        weights_derivatives = list(np.zeros((len(self.layers),)))

        loss = 0

        for example_input, example_label in zip(example_inputs, example_labels):
            if print_loss:
                loss += cost_f.evaluation(self.predict(example_input), example_label) / len(example_labels)
                # The += and len(...) is for automatically calculate the mean.

            output = self.predict(example_input, long_output=True)
            last_delta = None

            # Backpropagation
            for l, layer in enumerate(self.layers[::-1]):
                if l == 0:  # Last layer
                    last_delta = cost_f.deriv(output[-1][0], example_label) * layer.activation_func.deriv(output[-1][1])
                else:
                    last_delta = layer.W @ last_delta * \
                                 layer.activation_func.deriv(output[len(self.layers) - l - 1][1])

                # The += and len(example_labels) is to automatically calculate the mean.
                bias_derivatives[l] += last_delta / len(example_labels)
                weights_derivatives[l] += last_delta * output[len(self.layers) - l - 2][0] / len(example_labels)

        # Gradient descent
        for l in range(len(self.layers[::-1])):
            self.layers[len(self.layers) - l - 1].b -= lr * bias_derivatives[l]
            self.layers[len(self.layers) - l - 1].W -= lr * weights_derivatives[l]

        if print_loss:
            time_print(f"Epoch {epoch_number}, minibatch {minibatch_number}. Average loss: {loss}. ")


# For testing. This should be removed.
training_inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 1]
])

training_labels = np.array([
    [0, 1],
    [0, 0],
    [1, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])

test_inputs = np.array([
    [1, 1, 0],
    [1, 0, 1],
])

test_labels = np.array([
    [0, 1],
    [1, 0],
])


def sigmoid_ev(x):
    return 1 / (1 + np.e ** (-x))


sigmoid = Function(sigmoid_ev, lambda x: sigmoid_ev(x) * (1 - sigmoid_ev(x)))
mse = Function(lambda yp, yr: np.mean((yp - yr) ** 2), lambda yp, yr: (yp - yr))

network = Network([3, 3, 2], [sigmoid, sigmoid])
network.train(cost_f=mse, epochs=20, minibatch_size=3, lr=0.007, eval_func=(lambda x: x.round().all()),
              training_inputs=training_inputs, training_labels=training_labels, test_inputs=test_inputs,
              test_labels=test_labels)
