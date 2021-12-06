import numpy as np
import matplotlib.pyplot as plt


def sigmoid(i):
    return 1/(1 + np.exp(-i))


def get_dataset(set_name):
    import scipy.io
    m = scipy.io.loadmat('lab7/data.mat')
    data_set = m['data'][set_name][0][0]['inputs'][0][0]
    labels = m['data'][set_name][0][0]['targets'][0][0]
    return data_set, labels


inputs, targets = get_dataset('training')
targets_count = 10
inputs_ext = np.vstack([inputs, np.ones((1, 1000))])
w = np.random.normal(size=(10, 257)) / 257
epoch_errors = []
for i in range(1000):
    h = sigmoid(np.matmul(w, inputs_ext))
    errors = (h - targets)**2
    epoch_errors.append(np.mean(errors))
    derive_error = np.matmul((2 * (h-targets) * h * (1 - h) / 1000), inputs_ext.T)
    w -= derive_error * 0.2


#plt.plot(epoch_errors)
#plt.show()


test, targets = get_dataset('test')
test_ext = np.vstack([test, np.ones((1, 9000))])
result = sigmoid(np.matmul(w, test_ext))

digits = np.argmax(targets, axis=0)
pred_digits = np.argmax(result, axis=0)
print(np.sum(digits == pred_digits) / 9000)
