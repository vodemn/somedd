from bit_alloc import bit_alloc
from math import log2
import numpy as np


def quantize(img: np.ndarray, bloc_shape: tuple, init_b: int) -> np.ndarray:
    result = np.zeros(img.shape)
    bloc_h, bloc_w = bloc_shape

    bit_budget = bit_alloc(img, bloc_shape, init_b)
    for coord, bit in np.ndenumerate(bit_budget):
        h = 2 ** bit
        
        # coresponding coords in original array
        orig_y = coord[0] * bloc_h
        orig_x = coord[1] * bloc_w
        block_slice = (slice(orig_y, orig_y + bloc_h),
                       slice(orig_x, orig_x + bloc_w))

        for block_coord, value in np.ndenumerate(img[block_slice]):
            b, a = block_coord
            result[orig_y + b][orig_x + a] = np.round(value * h) / h

    return result
