import numpy as np


def get_count_variances(img: np.ndarray):
    variances = {}
    counts_dict = {}
    values, counts = np.unique(img, return_counts=True)
    for i, val in np.ndenumerate(values):
        counts_dict[val] = counts[i]
        variances[val] = counts[i] / img.size
    return variances, counts_dict


def histogram(variances: dict):
    hist = np.zeros(256)
    for val, variance in variances.items():
        hist[val] = variance
    return hist


def calc_entropy(variances: dict):
    entropy = 0
    for var in variances:
        entropy -= var * np.log2(var)
    return entropy


def avg_length(variances: dict, 
               huffman: dict):
    avg_len = 0
    for val, variance in variances.items():
        avg_len += variance * len(huffman[val])
    return avg_len


def calc_compression(encoded_img: np.ndarray, 
                huff_dict: dict):
    encoded = 0

    for _, val in np.ndenumerate(encoded_img):
        encoded += len(val)

    for var in huff_dict.values():
        encoded += 8 + len(var)

    return (8*encoded_img.size  / encoded) * 100 #percent
