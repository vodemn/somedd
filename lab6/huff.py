from cv2 import sort
import numpy as np

# https://www.w3resource.com/python-exercises/challenges/1/python-challenges-1-exercise-58.php

def huffmandict(counts_dict: dict):
    from heapq import heappush, heappop, heapify
    from collections import defaultdict

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

    for k, v in code.items():
        code[k] = ''.join(v)

    return dict(sorted(code.items(), key=lambda item: int(item[1], 2)))


def huffmanenco(img, huff_dict):
    result = np.empty(img.shape, dtype=object)
    for coord, val in np.ndenumerate(img):
        result[coord[0]][coord[1]] = huff_dict[val]
    return result


def huffmandeco(encoded_img: np.ndarray, huff_dict: dict):
    result = np.zeros(encoded_img.shape, dtype='int')
    reverse_huff = dict((v,k) for k,v in huff_dict.items())
    for coord, val in np.ndenumerate(encoded_img):
        result[coord[0]][coord[1]] = reverse_huff[val]
    return result