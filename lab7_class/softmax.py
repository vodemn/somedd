import numpy as np

def softmax(i):
    e = np.exp(i - np.max(i, axis=0, keepdims=True))
    e_sum = np.sum(e, axis = 0, keepdims=True)
    return e / e_sum