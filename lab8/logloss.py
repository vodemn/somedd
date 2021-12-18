import numpy as np

def logloss(target, result):
    return (-1 / target.shape[1]) * np.sum(target * np.log(result + 1e-5))

def logloss_derive(target, result):
    return result - target
