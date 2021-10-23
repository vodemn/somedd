from dct import dct2, dct2_inbuilt, idct2_inbuilt
from scipy.fftpack import dctn, idctn
from shuffle import shuffle
from split import split_in_blocks
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


def get_image(name):
    img = cv.imread(name)[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


img = get_image('lab4/lena_gray_256.tif')

h, w = img.shape
bloc_shape = (8, 8)
ibloc_shape = (h // 8, w // 8)

"""
img_dct = dct2(img, bloc_shape)
shuffeled = shuffle(img_dct, bloc_shape)
shuffeled_i = shuffle(shuffeled, ibloc_shape)

plt.subplot(2, 4, 1)
plt.imshow(img_dct, cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 4, 2)
plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 4, 3)
plt.imshow(shuffeled_i, cmap='gray', vmin=0, vmax=255)
"""

dct2 = lambda x: dctn(x, norm='ortho')
idct2 = lambda x: idctn(x, norm='ortho')

img_dct = split_in_blocks(img, (8, 8), dct2)
shuffeled = shuffle(img_dct, (32, 32))
shuffeled_i = shuffle(shuffeled, (8, 8))
img_idct = split_in_blocks(shuffeled_i, (8, 8), idct2)

plt.subplot(2, 3, 4)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 5)
plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
plt.subplot(2, 3, 6)
plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)

plt.show()
