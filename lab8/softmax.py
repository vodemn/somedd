import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    e_sum = np.sum(e, axis = 0, keepdims=True)
    return e / e_sum