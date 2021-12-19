import numpy as np


class Dense:

    def __init__(self, in_dim, neurons_count, h, h_derive, lr):
        self.weights = 2 * \
            np.random.random((neurons_count, in_dim + 1)) / np.sqrt(neurons_count + in_dim + 1)
        self.cache = None
        self.activation = h
        self.activation_derive = h_derive
        self.learning_rate = lr

    def forward(self, x):
        x_ext = np.vstack([x, np.ones((1, x.shape[1]))])
        O = self.activation(self.weights @ x_ext)
        self.cache = (x_ext, O)
        return O

    def backward(self, dE):
        input = self.cache[0]  # input from the previous layer      # (257, N)
        output = self.cache[1]  # recent result of this layer       # (10, N)
        deriv_act = dE                                              # (10, N)
        if self.activation_derive is not None:
            derived = self.activation_derive(output)
            deriv_act *= self.activation_derive(output)
        self.weights -= (deriv_act @ input.T) * self.learning_rate  # (10, 257)
        dE_next = self.weights.T @ deriv_act
        dE_next = dE_next[:dE_next.shape[0] - 1]
        return dE_next
