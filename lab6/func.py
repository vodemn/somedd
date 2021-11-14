from math import log2
import numpy as np


def count_variances(img):
    variances = {}
    counts_dict = {}
    values, counts = np.unique(img, return_counts=True)

    for index, value in np.ndenumerate(values):
        counts_dict[value] = counts[index]
        variances[value] = counts[index] / img.size
    return variances, counts_dict


def histogram(variances):
    hist = np.zeros(256)
    for k, v in variances.items():
        hist[k] = v
    return hist


def entropy(variances: list):
    ent = 0
    for val in variances:
        ent += val * log2(val)
    return -ent


def avg_length(variances: dict, huffman: dict):
    avg_len = 0
    for k, v in variances.items():
        avg_len += v * len(huffman[k])
    return avg_len


def compression(encoded_img: np.ndarray, huff_dict: dict):
    encoded = 0
    for _, val in np.ndenumerate(encoded_img):
        encoded += len(val)
    for k, v in huff_dict.items():
        encoded += 8 + len(v)
    return (encoded_img.size * 8 / encoded) * 100
