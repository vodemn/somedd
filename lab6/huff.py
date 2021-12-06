import numpy as np

from heapq import heappush, heappop
from collections import defaultdict


def get_huff_dict(counts_dict: dict):
    code = defaultdict(list)
    heap = [(count, [value]) for value, count in counts_dict.items()]

    while len(heap) > 1:
        freq0, letters0 = heappop(heap)
        for ltr in letters0:
            code[ltr].insert(0, '0')
        freq1, letters1 = heappop(heap)
        for ltr in letters1:
            code[ltr].insert(0, '1')
        heappush(heap, (freq0+freq1, letters0+letters1))

    for key, val in code.items():
        code[key] = ''.join(val)

    return dict(sorted(code.items(), key=lambda x: int(x[1], 2)))


def huff_encode(img, huff_dict):
    result = np.empty(img.shape, dtype=object)
    for (x, y), val in np.ndenumerate(img):
        result[x][y] = huff_dict[val]
    return result


def huff_decode(encoded_img: np.ndarray, huff_dict: dict):
    result = np.zeros(encoded_img.shape, dtype='int')
    reverse_huff = dict((v,k) for k,v in huff_dict.items())
    for (x, y), val in np.ndenumerate(encoded_img):
        result[x][y] = reverse_huff[val]
    return result