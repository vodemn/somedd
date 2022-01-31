import numpy as np
import matplotlib.pyplot as plt

from dense import Dense
from sigmoid import sigmoid, sigmoid_derive


def get_dataset(set_name):
    import scipy.io
    m = scipy.io.loadmat('lab7/data.mat')
    data_set = m['data'][set_name][0][0]['inputs'][0][0]
    labels = m['data'][set_name][0][0]['targets'][0][0]
    return data_set, labels

layer = Dense(256, 10, sigmoid, sigmoid_derive, 0.2)

inputs, targets = get_dataset('training')
epoch_errors = []
for i in range(1000):
    h = layer.forward(inputs)
    errors = (h - targets)**2
    epoch_errors.append(np.mean(errors))
    layer.backward(h - targets)

plt.plot(epoch_errors)
plt.show()


test, targets = get_dataset('test')
result = layer.forward(test)

digits = np.argmax(targets, axis=0)
pred_digits = np.argmax(result, axis=0)
print(np.sum(digits == pred_digits) / 9000)
