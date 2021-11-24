from math import log2
import numpy as np

epsilon = 1/12


def __round_bit(init_b, variance, divider):
    result = init_b + 0.5 * log2(variance / divider)
    return 0 if (result <= 0) else np.round(result, 0)


def bit_alloc(img, bloc_shape, init_b) -> np.ndarray:
    bloc_h, bloc_w = bloc_shape
    img_var = np.zeros((img.shape[0] // bloc_h, img.shape[1] // bloc_w))

    for coord, _ in np.ndenumerate(img_var):
        y, x = coord
        block_slice = (slice(y * bloc_h, y * bloc_h + bloc_h),
                       slice(x * bloc_w, x * bloc_w + bloc_w))

        img_var[y][x] = epsilon * np.var(img[block_slice])

    divider = np.prod(np.power(img_var, 1 / img_var.size))
    return np.array([[__round_bit(init_b, i, divider) for i in row] for row in img_var])
