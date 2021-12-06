import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from huffman import codebook
from huff import get_huff_dict, huff_encode, huff_decode
from utility import (get_count_variances, histogram, 
                  calc_entropy, avg_length, calc_compression)


img_name = './test_images/kodim08.png'
img = cv.cvtColor(cv.imread(img_name)
                  [..., ::-1][:256, :256], cv.COLOR_BGR2GRAY)

variances, counts_dict = get_count_variances(img)
hist = histogram(variances)

plt.bar(np.arange(256),  hist)
plt.show()

ent = calc_entropy(variances.values())
print('Entropy: ' + str(ent))

print('\nOwn huffdict')
huff_dict = get_huff_dict(counts_dict)
encoded_img = huff_encode(img, huff_dict)

avg_len = avg_length(variances, huff_dict)
print('Average length: ' + str(avg_len))

decoded_img = huff_decode(encoded_img, huff_dict)
plt.imshow(decoded_img, cmap='gray', vmin=0, vmax=255)
plt.show()

compression_coef = calc_compression(encoded_img, huff_dict)
print('Compressed coefficient: ' + str(compression_coef) + '%')

# Package huff_dict
print('\nPython huffdict')
huff_dict = codebook(counts_dict.items())
encoded_img = huff_encode(img, huff_dict)

avg_len = avg_length(variances, huff_dict)
print('Average length: ' + str(avg_len))

decoded_img = huff_decode(encoded_img, huff_dict)
plt.imshow(decoded_img, cmap='gray', vmin=0, vmax=255)
plt.show()

compression_coef = calc_compression(encoded_img, huff_dict)
print('Compressed coefficient: ' + str(compression_coef) + '%')

