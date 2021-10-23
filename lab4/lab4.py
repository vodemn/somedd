from dct import dct2, idct2
from scipy.fftpack import dctn, idctn
from shuffle import shuffle
from split import split_in_blocks
import cv2 as cv
import matplotlib.pyplot as plt



def get_image(name):
    img = cv.imread(name)[..., ::-1]
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


img = get_image('lab4/lena_gray_256.tif')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

h, w = img.shape
bloc_shape = (8, 8)
ibloc_shape = (h // 8, w // 8)

img_dct = split_in_blocks(img, (8, 8), dct2)
shuffeled = shuffle(img_dct, (32, 32))
img_idct = split_in_blocks(shuffle(shuffeled, (8, 8)), (8, 8), idct2)

plt.subplot(1, 2, 1)
plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)
plt.show()

def lambda_dct2(x): return dctn(x, norm='ortho')
def lambda_idct2(x): return idctn(x, norm='ortho')
img_dct = split_in_blocks(img, bloc_shape, lambda_dct2)
shuffeled = shuffle(img_dct, ibloc_shape)
shuffeled_i = shuffle(shuffeled, bloc_shape)
img_idct = split_in_blocks(shuffeled_i, bloc_shape, lambda_idct2)

plt.subplot(1, 2, 1)
plt.imshow(shuffeled, cmap='gray', vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.imshow(img_idct, cmap='gray', vmin=0, vmax=255)
plt.show()
