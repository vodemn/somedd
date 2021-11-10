import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn


def alpha(x, n):
    return np.sqrt((1 if x == 0 else 2) / n)

def dct2(block) -> np.ndarray:
    """DCT2 realization using numpy"""
    M, N = block.shape
    res = np.zeros(block.shape)

    for (p, q), _ in np.ndenumerate(res):
        for (m, n), _ in np.ndenumerate(block):
            cos_mp = np.cos(np.pi * (2*m + 1) * p / (2*M))
            cos_np = np.cos(np.pi * (2*n + 1) * q / (2*M))
            res[p][q] += block[m][n] * cos_mp * cos_np
        res[p][q] *= alpha(p, M) * alpha(q, N)

    return res


def idct2(block) -> np.ndarray:
    """IDCT2 realization using numpy"""
    M, N = block.shape
    res = np.zeros(block.shape)

    for (m, n), _ in np.ndenumerate(res):
        for (p, q), _ in np.ndenumerate(block):
            cos_mp = np.cos(np.pi * (2*m + 1) * p / (2*M))
            cos_np = np.cos(np.pi * (2*n + 1) * q / (2*M))
            res[m][n] += alpha(p, M) * alpha(q, N) * \
                         block[p][q] * cos_mp * cos_np

    return res
