import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import huffman as hf

from math import log10, sqrt
from scipy.fftpack import dctn, idctn

from compress import compress
from quantize import quantize
from shuffle import shuffle
from split import split


def read_image(name):
    img = cv.cvtColor(cv.imread(name)[..., ::-1][:256, :256], cv.COLOR_BGR2GRAY)
    return cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


init_bs = np.arange(1,9)
compressed_bits_result = np.zeros(init_bs.shape)
psnr_result = np.zeros(init_bs.shape)

img = read_image('./test_images/kodim08.png')
M, N = img.shape
bloc_shape = (8, 8)
ibloc_shape = (M // 8, N // 8)


def main():
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()

    for b in init_bs:
        shuffeled = shuffle(split(img, bloc_shape, 
                                  lambda x: dctn(x, norm='ortho')), 
                            ibloc_shape)
        quantized = quantize(shuffeled, ibloc_shape, b)

        compressed = split(shuffle(quantized, bloc_shape), 
                           bloc_shape, lambda x: idctn(x, norm='ortho'))
        psnr_result[b - 1] = psnr(img, compressed)
        plt.imsave('./lab5/result_images/' + str(b) + '.png', 
                   compressed, cmap='gray', vmin = 0, vmax = 1)

        encoded, codes = compress(quantized, ibloc_shape)
        compressed_bits = 0
        for _, val in np.ndenumerate(encoded):
            compressed_bits += val.item().bit_length()

        for code in codes:
            for _, v in code.items():
                compressed_bits += v.bit_length() + 32
        
        compressed_bits_result[b - 1] = compressed_bits

    plt.plot(compressed_bits_result, psnr_result)
    plt.show()


if __name__ == '__main__':
    main()