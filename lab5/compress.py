from rle import run_length_encoding
from zig_zag import zig_zag
from huffman import huffman

import numpy as np


def compress(img, bloc_shape):
    h, w = img.shape
    bloc_h, bloc_w = bloc_shape
    result = np.zeros(img.shape, dtype='int')
    codes = []

    for coord, _ in np.ndenumerate(np.zeros((h // bloc_h, w // bloc_w))):
        # size of resulting block
        block_slice = (slice(bloc_h * coord[0], bloc_h * (coord[0] + 1)),
                       slice(bloc_w * coord[1], bloc_w * (coord[1] + 1)))

        line = run_length_encoding(zig_zag(img[block_slice]))
        huff_dict = huffman.codebook(line)

        block_codes = {}
        for k, v in huff_dict.items():
            block_codes[k] = int('0' if v == '' else v, 2)

        codes.append(block_codes)
        for block_coord, val in np.ndenumerate(img[block_slice]):
            result[block_slice][block_coord] = block_codes[val]

    return result, codes
