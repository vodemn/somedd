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
w = np.random.uniform(-np.sqrt(6/(10+256)), np.sqrt(6/(10+256)), (10, 256))
epoch_errors = []
for i in range(100):
    h = sigmoid(np.matmul(w, inputs) + np.zeros(targets_count)[:, None])
    errors = (h - targets)**2
    epoch_errors.append(np.mean(errors))
    derive_error = np.matmul((2 * (h-targets) * h * (1 - h) / 1000), inputs.T)
    w -= derive_error * 0.2


plt.plot(epoch_errors)
plt.show()


test, targets = get_dataset('test')
result = sigmoid(np.matmul(w, test) + np.zeros(targets_count)[:, None])

digits = np.argmax(targets, axis=0)
pred_digits = np.argmax(result, axis=0)
print(np.sum(digits == pred_digits))
