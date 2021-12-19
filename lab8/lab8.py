import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from dense import Dense
from logloss import logloss, logloss_derive
from sigmoid import sigmoid, sigmoid_derive
from softmax import softmax


def int_to_onehot(n: int) -> np.array:
    v = [0] * 10
    v[n] = 1
    return np.array(v)


def run_lab(lr, first_layer_size: int = 256, show_plot: bool = True):
    m = scipy.io.loadmat('lab8/dataset.mat')
    inputs = m['inputs']
    targets = np.array([int_to_onehot(number[0]) for number in m['targets'].T]).T

    bigger_part = int(inputs.shape[1] * 0.8)

    training_inputs = inputs[..., :bigger_part]
    training_targets = targets[..., :bigger_part]
    test_inputs = inputs[..., bigger_part:]
    test_targets = targets[..., bigger_part:]

    layers = [Dense(training_inputs.shape[0], first_layer_size, sigmoid, sigmoid_derive, lr),
              Dense(first_layer_size, training_targets.shape[0], softmax, None, lr)]

    epoch_errors = []
    for i in range(100):
        result = training_inputs
        for layer in layers:
            result = layer.forward(result)

        loss = logloss(training_targets, result)
        epoch_errors.append(loss)

        dE = logloss_derive(training_targets, result)
        for layer in list(reversed(layers)):
            dE = layer.backward(dE)

    if (show_plot):
        plt.plot(epoch_errors)
        plt.show()

    result = test_inputs
    for layer in layers:
        result = layer.forward(result)

    digits = np.argmax(test_targets, axis=0)
    predicted_digits = np.argmax(result, axis=0)
    prediction_accuracy = np.sum(digits == predicted_digits) / test_inputs.shape[1]
    return prediction_accuracy
