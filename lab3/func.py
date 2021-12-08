import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# common
def __pad_image(img, area_size):
    area_r = area_size // 2
    return np.pad(img, (area_r, area_r), 'constant',
                  constant_values=(0, 0))


def get_image(name):
    img = cv.imread('lab3/test_images/' + name + '.tif')[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def __base_filter(img, area_size, filter):
    result_img = np.zeros(img.shape)
    padded_img = __pad_image(img, area_size)

    h, w = img.shape
    area_r = area_size // 2
    for coord, v in np.ndenumerate(padded_img[area_r:h + area_r, area_r:w + area_r]):
        i, j = coord
        area = padded_img[i:i + 2 * area_r + 1, j:j + 2 * area_r + 1]
        result_img[i][j] = filter(area)
    return result_img


# 3.1
__brightness_levels = 256
ref = np.arange(__brightness_levels)


def count_p(img):
    probabilities = {}
    values, counts = np.unique(img, return_counts=True)
    for index, value in np.ndenumerate(values):
        probabilities[value] = counts[index] / img.size
    return probabilities


def histogram(img):
    hist = np.zeros(ref.size)
    for k, v in count_p(img).items():
        hist[k] = v
    return hist

def equalize(img):
    p = count_p(img)
    t = np.cumsum(np.fromiter(p.values(), dtype='float'), 0) * \
        (__brightness_levels - 1)
    new_p = dict(zip(p.keys(), t))
    return [[new_p[value] for value in row] for row in img]


# 3.2
def __equalize_local(area):
    area_h, area_w = area.shape
    return equalize(area)[area_h][area_h]


def equalize_local(img, area_size):
    result_img = np.zeros(img.shape)
    padded_img = __pad_image(img, area_size)

    h, w = img.shape
    area_r = area_size // 2
    for coord, v in np.ndenumerate(padded_img[area_r:h + area_r, area_r:w + area_r]):
        i, j = coord
        area = padded_img[i:i + 2 * area_r + 1, j:j + 2 * area_r + 1]
        result_img[i][j] = equalize(area)[area_r][area_r]

    return result_img


# 3.3
def __flat_filter(s, t, m, n):
    a = (m - 1)/2
    b = (n - 1)/2
    if -a <= s <= a and -b <= t <= b:
        return 1/(m*n)
    else:
        return 0


def __convulate(area):
    m, n = area.shape
    result = 0
    for coord, v in np.ndenumerate(area):
        s, t = coord
        result += __flat_filter(s, t, m, n) * area[m - s - 1][n - t - 1]
    return result


def my_2dfilter(img, area_size):
    return __base_filter(img, area_size, __convulate)


def threshold_filter(img, threshold):
    result_img = np.zeros(img.shape)

    h, w = img.shape
    for coord, v in np.ndenumerate(img):
        i, j = coord
        if (img[i][j] > 64):
            result_img[i][j] = 255
    return result_img


# 3.4
def median_filter(img, area_size):
    median = lambda x: np.median(np.concatenate(x), axis=0)
    return __base_filter(img, area_size, median)


# 3.5
def __laplasian(area):
    __coeffs = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    return np.sum(np.concatenate(area * __coeffs), axis=0)


def laplasian_filter_3x3(img):
    return __base_filter(img, 3, __laplasian)


def remove_back(img, filtered_img):
    return np.subtract(img, filtered_img)
