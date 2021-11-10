from cv2 import sort
import numpy as np

# https://www.w3resource.com/python-exercises/challenges/1/python-challenges-1-exercise-58.php


def huffmandict(counts_dict: dict):
    from heapq import heappush, heappop, heapify
    from collections import defaultdict

    # mapping of letters to codes
    code = defaultdict(list)

    # Using a heap makes it easy to pull items with lowest frequency.
    # Items in the heap are tuples containing a list of letters and the
    # combined frequencies of the letters in the list.
    heap = [(count, [value]) for value, count in counts_dict.items()]

    # Reduce the heap to a single item by combining the two items
    # with the lowest frequencies.
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
    result = np.zeros(img.shape, dtype='int')
    for coord, val in np.ndenumerate(img):
        result[coord[0]][coord[1]] = int('0b' + huff_dict[val], 2)
    return result
