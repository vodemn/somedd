import cv2 as cv
import matplotlib.pyplot as plt

from dct import dct2, idct2
from scipy.fftpack import dctn, idctn
from shuffle import shuffle
from split import split


def read_image(name):
    img = cv.imread(name)[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = read_image('./lena_gray_256.tif')
M, N = img.shape
block_shape = (8, 8)
iblock_shape = (M // 8, N // 8)


def my_realization():
    img_dct = split(img, block_shape, dct2)
    shuffeled = shuffle(img_dct, iblock_shape)
    img_idct = split(shuffle(shuffeled, block_shape), block_shape, idct2)

    plt.figure("My realization")
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)
    

def python_realization():
    img_dct = split(img, block_shape, lambda x: dctn(x, norm='ortho'))
    shuffeled = shuffle(img_dct, iblock_shape)
    shuffeled_i = shuffle(shuffeled, block_shape)
    img_idct = split(shuffeled_i, block_shape, lambda x: idctn(x, norm='ortho'))

    plt.figure("Python realization")
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)
    

if __name__ == '__main__':
    my_realization()
    python_realization()
    plt.show()