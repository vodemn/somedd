from bit_alloc import bit_alloc
from math import log2
import numpy as np


def quantize(img, bloc_shape, init_b) -> np.ndarray:
    h, w = img.shape
    result = np.zeros(img.shape)
    bloc_h, bloc_w = bloc_shape

    bit_budget = bit_alloc(img, bloc_shape, init_b)
    bit_budget = [[2 ** bit for bit in row] for row in bit_budget]

    for coord, bit in np.ndenumerate(bit_budget):
        # coresponding coords in original array
        orig_y = coord[0] * bloc_h
        orig_x = coord[1] * bloc_w

        # size of resulting block
        block_slice = (slice(orig_y, orig_y + bloc_h),
                       slice(orig_x, orig_x + bloc_w))

        for block_coord, value in np.ndenumerate(img[block_slice]):
            b, a = block_coord
            result[orig_y + b][orig_x + a] = np.round(value * bit) / bit

    return result
