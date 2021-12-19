import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from dense import Dense
from logloss import logloss, logloss_derive
from sigmoid import sigmoid, sigmoid_derive
from softmax import softmax


def int_to_onehot(n) -> np.array:
    v = [0] * 10
    v[n] = 1
    return np.array(v)


m = scipy.io.loadmat('lab8/dataset.mat')
inputs = m['inputs']
targets = np.array([int_to_onehot(number[0]) for number in m['targets'].T]).T

bigger_part = int(inputs.shape[1] * 0.8)

training_inputs = inputs[..., :bigger_part]
training_targets = targets[..., :bigger_part]
test_inputs = inputs[..., bigger_part:]
test_targets = targets[..., bigger_part:]

print(training_inputs.shape, training_targets.shape,
      test_inputs.shape, test_targets.shape)


layers = [Dense(256, 256, sigmoid, sigmoid_derive, 0.3),
          Dense(256, 10, softmax, None, 0.3)]

epoch_errors = []
for i in range(100):
    # get result
    # TODO понять, почему ошибка не уменьшается
    layer1_out = layers[0].forward(training_inputs)
    layer2_out = layers[1].forward(layer1_out)
    result = layer2_out

    # error
    loss = logloss(training_targets, result)
    epoch_errors.append(loss)

    dZ = logloss_derive(training_targets, result)
    for layer in list(reversed(layers)):
        dZ = layer.backward(dZ)

plt.plot(epoch_errors)
plt.show()

