import numpy as np

from rle import run_length_encoding
from zig_zag import zig_zag
from huffman import huffman


def compress(img, block_shape):
    h, w = img.shape
    block_H, block_W = block_shape
    result = np.zeros(img.shape, dtype='int')
    codes = []

    for (m, n), _ in np.ndenumerate(np.zeros((h // block_H, w // block_W))):
        # size of resulting block
        block_window = (slice(m*block_H, (m+1) * block_H),
                       slice(n*block_W, (n+1) * block_W))

        line = run_length_encoding(zig_zag(img[block_window]))
        huff_dict = huffman.codebook(line)

        block_codes = {}
        for k, v in huff_dict.items():
            block_codes[k] = int('0' if v == '' else v, 2)

        codes.append(block_codes)
        for block_crd, val in np.ndenumerate(img[block_window]):
            result[block_window][block_crd] = block_codes[val]

    return result, codes
