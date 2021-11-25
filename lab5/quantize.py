import numpy as np
from math import log2
from bit_alloc import bit_alloc


def quantize(img: np.ndarray, 
             block_shape: tuple, 
             init_b: int) -> np.ndarray:
    result = np.zeros(img.shape)
    block_H, block_W = block_shape

    bit_val = bit_alloc(img, block_shape, init_b)
    for (y, x), bit in np.ndenumerate(bit_val):
        h = 2 ** bit
        # getting coordinatess in image
        img_y = y * block_H
        img_x = x * block_W
        block_window = (slice(img_y, img_y + block_H),
                        slice(img_x, img_x + block_W))

        for (b, a), val in np.ndenumerate(img[block_window]):
            result[img_y + b][img_x + a] = np.round(val * h) / h

    return result
