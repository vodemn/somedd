from math import log2
import numpy as np


def _round_bit(init_b: int, 
               variance: float, 
               divider: float):
    result = init_b + 0.5 * log2(variance / divider)
    return 0 if (result <= 0) else np.round(result, 0)


def bit_alloc(img: np.ndarray, 
              block_shape: tuple, 
              init_b: int, 
              eps: float=1/12) -> np.ndarray:
    block_H, block_W = block_shape
    img_var = np.zeros((img.shape[0] // block_H, img.shape[1] // block_W))

    for (m, n), _ in np.ndenumerate(img_var):
        window_slice = (slice(m*block_H, (m+1) * block_H),
                        slice(n*block_W, (n+1) * block_W))
        img_var[m][n] = eps * (np.var(img[window_slice]))

    divider = np.prod(img_var ** (1 / img_var.size))
    return np.array(
        [[_round_bit(init_b, var, divider) 
            for var in row] 
                for row in img_var])
