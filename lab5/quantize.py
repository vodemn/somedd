from bit_alloc import bit_alloc
from math import log2
import numpy as np


def quantize(img, bloc_shape, init_b) -> np.ndarray:
    h, w = img.shape
    result = np.zeros(img.shape)
    bloc_h, bloc_w = bloc_shape

    bit_budget = bit_alloc(img, bloc_shape, init_b)

    for coord, bit in np.ndenumerate(bit_budget):
        y, x = coord

        # coresponding coords in original array
        orig_y = y * bloc_h
        orig_x = x * bloc_w

        # size of resulting block
        block_slice = (slice(orig_y, orig_y + bloc_h),
                       slice(orig_x, orig_x + bloc_w))

        h = 2 ** bit

        for coord, value in np.ndenumerate(img[block_slice]):
            result[block_slice][coord[0]][coord[1]] = np.round(value * h) / h

    return result
