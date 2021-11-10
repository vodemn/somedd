from math import log2
import numpy as np

def entropy(img, p):
    ent = 0
    for _, val in np.ndenumerate(img):
        ent += p[val] * log2(p[val])
    return -ent