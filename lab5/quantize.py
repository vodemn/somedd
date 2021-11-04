from bit_alloc import bit_alloc
from math import log2
import numpy as np


def quantize(img, bloc_shape, init_b) -> np.ndarray:
    h, w = img.shape
    result = np.zeros(img.shape)
    bloc_h, bloc_w = bloc_shape

    bit_budget = np.round(bit_alloc(img, bloc_shape, init_b), 0)
    bit_budget = [[0 if i <= 0 else i for i in row] for row in bit_budget]

    # size of resulting 2d array of block
    result_h = h // bloc_h
    result_w = w // bloc_w

    for i in range(result_h * result_w):
        # coords of a block in 2d array of blocks
        y = i // result_w
        x = i - y * result_w

        # coresponding coords in original array
        orig_y = y * bloc_h
        orig_x = x * bloc_w

        # size of resulting block
        y_slice = slice(orig_y, orig_y + bloc_h)
        x_slice = slice(orig_x, orig_x + bloc_w)

        h = 2 ** bit_budget[y][x]

        for coord, value in np.ndenumerate(img[y_slice, x_slice]):
            result[y_slice, x_slice][coord[0]][coord[1]] = np.round(value * h) / h

        return result
