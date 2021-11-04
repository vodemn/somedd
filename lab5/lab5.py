from quantize import quantize
from shuffle import shuffle
from split import split_in_blocks
from math import log10, sqrt
from scipy.fftpack import dctn, idctn
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_image(name):
    img = cv.imread(name)[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

init_bs = np.arange(1,9)
psnr_result = np.zeros(init_bs.shape)

img = get_image('lab5/lena_gray_256.tif')

h, w = img.shape
bloc_shape = (8, 8)
ibloc_shape = (h // 8, w // 8)


def dct2(x): return dctn(x, norm='ortho')
def idct2(x): return idctn(x, norm='ortho')

for b in init_bs:
    shuffeled = shuffle(split_in_blocks(img, bloc_shape, dct2), ibloc_shape)
    quantized = quantize(shuffeled, ibloc_shape, 4)
    compressed = split_in_blocks(shuffle(quantized, bloc_shape), bloc_shape, idct2)
    psnr_result[b - 1] = psnr(img, compressed)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(compressed, cmap='gray', vmin=0, vmax=255)
    plt.show()

