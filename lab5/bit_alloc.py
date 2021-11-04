from math import log2
import numpy as np

epsilon = 1/12

def __round_bit(init_b, variance, divider):
    result = init_b + 0.5 * log2(variance / divider)
    if (result <= 0):
        return 0
    else:
        return np.round(result, 0)

def bit_alloc(img, bloc_shape, init_b) -> np.ndarray:
    h, w = img.shape
    bloc_h, bloc_w = bloc_shape

    # size of resulting 2d array of block
    result_h = h // bloc_h
    result_w = w // bloc_w

    img_var = np.zeros((result_h, result_w))

    for i in range(result_h * result_w):
        # coords of a block in 2d array of blocks
        y = i // result_w
        x = i - y * result_w

        # coresponding coords in original array
        orig_y = y * bloc_h
        orig_x = x * bloc_w

        # size of resulting block
        block_slice = (slice(orig_y, orig_y + bloc_h),
                       slice(orig_x, orig_x + bloc_w))

        img_var[y][x] = epsilon * (np.var(img[block_slice]) ** 2)

    divider = np.prod(np.power(img_var, 1 / img_var.size))
    return np.array([[__round_bit(init_b, i, divider) for i in row] for row in img_var])
