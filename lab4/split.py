import numpy as np
import time


def split_in_blocks(img, bloc_shape, block_processor) -> np.ndarray:
    start_time = time.time()
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

        # coresponding coords in original array
        orig_y = y * bloc_h
        orig_x = x * bloc_w

        # size of resulting block
        y_slice = slice(orig_y, orig_y + bloc_h)
        x_slice = slice(orig_x, orig_x + bloc_w)
        result[y_slice, x_slice] = block_processor(img[y_slice, x_slice])
    print("%ss" % (time.time() - start_time))
    return result
