import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from huffman import codebook
from huff import huffmandict, huffmanenco, huffmandeco
from func import count_variances, histogram, entropy, avg_length, compression


img_name = 'test_images/kodim14.png'
img = cv.cvtColor(cv.imread(img_name)
                  [..., ::-1][:256, :256], cv.COLOR_BGR2GRAY)

variances, counts_dict = count_variances(img)
hist = histogram(variances)

plt.bar(np.arange(256),  hist)
plt.show()

ent = entropy(variances.values())
print('Entropy: ' + str(ent))

print('\nSelf-writtem huffdict')
huff_dict = huffmandict(counts_dict)
encoded_img = huffmanenco(img, huff_dict)

avg_len = avg_length(variances, huff_dict)
print('Avg length: ' + str(avg_len))

decoded_img = huffmandeco(encoded_img, huff_dict)
plt.imshow(decoded_img, cmap='gray', vmin=0, vmax=255)
plt.show()

compression_coef = compression(encoded_img, huff_dict)
print('Compressed: ' + str(compression_coef) + '%')

# Package huffmandict
print('\nPackage huffdict')
huff_dict = codebook(counts_dict.items())
encoded_img = huffmanenco(img, huff_dict)

avg_len = avg_length(variances, huff_dict)
print('Avg length: ' + str(avg_len))

decoded_img = huffmandeco(encoded_img, huff_dict)
plt.imshow(decoded_img, cmap='gray', vmin=0, vmax=255)
plt.show()

compression_coef = compression(encoded_img, huff_dict)
print('Compressed: ' + str(compression_coef) + '%')
