from rle import run_length_encoding
from zig_zag import zig_zag
from huffman import huffman

import numpy as np


def compress(img, bloc_shape):
    h, w = img.shape
    bloc_h, bloc_w = bloc_shape
    result = np.zeros(img.shape, dtype='int')
    codes = []
    bit_count = 0

    for coord, _ in np.ndenumerate(np.zeros((h // bloc_h, w // bloc_w))):
        y, x = coord

        # coresponding coords in original array
        orig_y = y * bloc_h
        orig_x = x * bloc_w

        # size of resulting block
        block_slice = (slice(orig_y, orig_y + bloc_h),
                       slice(orig_x, orig_x + bloc_w))
        
        line = run_length_encoding(zig_zag(img[block_slice]))
        huff_dict = huffman.codebook(line)

        for k, v in huff_dict.items():
            huff_dict[k] = int(v, 2)
            bit_count += int(v, 2).bit_length()  # bit length of encoded value
            bit_count += 8  # bit value of original value

        codes.append(huff_dict)
        for block_coord, val in np.ndenumerate(img[block_slice]):
            bit_count += huff_dict[val].bit_length()
            result[block_slice][block_coord] = huff_dict[val]

    return result, codes, bit_count
