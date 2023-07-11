import numpy as np

def sigmoid(i):
    return 1/(1 + np.exp(-i))

def sigmoid_derive(i):
    return i * (1 - i)
    #return sigmoid(i) * (1 - sigmoid(i))