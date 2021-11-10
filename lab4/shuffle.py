import numpy as np
from typing import Tuple


def shuffle(img: np.ndarray, 
            block_shape: Tuple) -> np.ndarray:
    M, N = img.shape
    res = np.zeros(img.shape)
    block_M, block_N = block_shape
    # size of resulting 2d array of block
    res_M = M // block_M
    res_N = N // block_N

    for y in range(res_M):
        for x in reversed(range(res_N)):
            y_ = y * block_M
            x_ = x * block_N
            # size of resulting block
            x_slice = slice(x_, x_ + block_N)
            y_slice = slice(y_, y_ + block_M)
            res[y_slice, x_slice] = img[y::res_M, x::res_N]
    
    return res
