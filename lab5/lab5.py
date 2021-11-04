from bit_budget import bit_budget
from quantize import quantize
from shuffle import shuffle
from split import split_in_blocks
from scipy.fftpack import dctn, idctn
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_image(name):
    img = cv.imread(name)[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


img = get_image('lab5/lena_gray_256.tif')

h, w = img.shape
bloc_shape = (8, 8)
ibloc_shape = (h // 8, w // 8)

def lambda_dct2(x): return dctn(x, norm='ortho')
def lambda_idct2(x): return idctn(x, norm='ortho')

img_dct = split_in_blocks(img, bloc_shape, lambda_dct2)
shuffeled = shuffle(img_dct, ibloc_shape)
quantized = quantize(shuffeled, ibloc_shape, bit_budget(shuffeled, ibloc_shape, 4))
shuffeled_i = shuffle(quantized, bloc_shape)
img_idct = split_in_blocks(shuffeled_i, bloc_shape, lambda_idct2)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)
plt.show()
