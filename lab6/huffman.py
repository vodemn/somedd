from cv2 import sort
import numpy as np


def huffmandict(img):
    values, counts = np.unique(img, return_counts=True)

    inds = np.flip(counts.argsort())
    values = values[inds]
    counts = counts[inds]
    print(values, counts)

huffmandict(['a', 'a', 'b', 'b', 'b', 'b', 'c'])