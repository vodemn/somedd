import numpy as np
import matplotlib.pyplot as plt


def sigmoid(i):
    return 1/(1 + np.exp(-i))


def get_train_test_data(set_name):
    import scipy.io
    m = scipy.io.loadmat('./lab7/data.mat')
    data_set = m['data'][set_name][0][0]['inputs'][0][0]
    labels = m['data'][set_name][0][0]['targets'][0][0]
    return data_set, labels

def get_test_loss(test, targets):
    linear = np.matmul(weights, test) + bias
    score = sigmoid(linear)

    digits = np.argmax(targets, axis=0)
    pred_digits = np.argmax(score, axis=0)
    return np.sum(digits == pred_digits) / len(digits)


inputs, targets = get_train_test_data('training')
test, test_targets = get_train_test_data('test')
out_features = 10
lr = 2e-4
np.random.seed(0)
weights = np.random.uniform(-np.sqrt(6/(10+256)), np.sqrt(6/(10+256)), (10, 256))
bias = np.random.uniform(-np.sqrt(6/(10+256)), np.sqrt(6/(10+256)), (10, 1))
epoch_loss = []
test_loss = []
for i in range(1500):
    linear = np.matmul(weights, inputs) + bias
    score = sigmoid(linear)

    loss = (score - targets) ** 2
    epoch_loss.append(np.mean(loss))
    w_grad = np.matmul((2 * (score-targets) * score * (1 - score)), inputs.T)
    b_grad = np.sum(w_grad, axis=1, keepdims=True)
    weights -= w_grad * lr
    bias -= b_grad * lr

    test_loss.append(get_test_loss(test, test_targets))


plt.subplot(1, 2, 1)
plt.plot(epoch_loss)
plt.subplot(1, 2, 2)
plt.plot(test_loss)
plt.show()
