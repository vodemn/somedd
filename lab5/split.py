import numpy as np


def split_in_blocks(img: np.ndarray, bloc_shape: tuple, block_processor) -> np.ndarray:
    h, w = img.shape
    result = np.zeros(img.shape)

    bloc_h, bloc_w = bloc_shape

    # size of resulting 2d array of block
    result_h = h // bloc_h
    result_w = w // bloc_w

    for i in range(result_h * result_w):
        # coords of a block in 2d array of blocks
        y = i // result_w
        x = i - y * result_w

        # size of resulting block
        block_slice = (slice(bloc_h * y, bloc_h * (y + 1)),
                       slice(bloc_w * x, bloc_w * (x + 1)))

        result[block_slice] = block_processor(img[block_slice])
    return result
