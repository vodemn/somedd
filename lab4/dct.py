import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn


def a(x, n):
    return math.sqrt((1 if x == 0 else 2)/n)

def dct2(bloc) -> np.ndarray:
    """Applies dct to the selected bloc of image"""
    M, N = bloc.shape
    result_bloc = np.zeros(bloc.shape)

    for coord, v in np.ndenumerate(result_bloc):
        p, q = coord
        for coord, v in np.ndenumerate(bloc):
            m, n = coord
            cos_mp = math.cos(math.pi * (2 * m + 1) * p / (2 * M))
            cos_nq = math.cos(math.pi * (2 * n + 1) * q / (2 * M))
            result_bloc[p][q] += bloc[m][n] * cos_mp * cos_nq
        result_bloc[p][q] *= a(p, M) * a(q, N)

    return result_bloc


def idct2(bloc) -> np.ndarray:
    """Applies idct to the selected bloc of image"""
    M, N = bloc.shape
    result_bloc = np.zeros(bloc.shape)

    for coord, v in np.ndenumerate(result_bloc):
        m, n = coord
        for coord, v in np.ndenumerate(bloc):
            p, q = coord
            cos_mp = math.cos(math.pi * (2 * m + 1) * p / (2 * M))
            cos_nq = math.cos(math.pi * (2 * n + 1) * q / (2 * M))
            result_bloc[m][n] += a(p, M) * a(q, N) * bloc[p][q] * cos_mp * cos_nq

    return result_bloc
