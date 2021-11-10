import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from huffman import huffmandict, huffmanenco
from entropy import entropy


img_name = 'test_images/kodim14.png'
img = cv.cvtColor(cv.imread(img_name)
                  [..., ::-1][:256, :256], cv.COLOR_BGR2GRAY)

ref = np.arange(256)

variances = {}
huff_dict = {}
values, counts = np.unique(img, return_counts=True)
for index, value in np.ndenumerate(values):
    huff_dict[value] = counts[index]
    variances[value] = counts[index] / img.size

hist = np.zeros(ref.size)
for k, v in variances.items():
    hist[k] = v

#plt.subplot(1, 2, 1)
#plt.bar(ref,  hist)
#plt.subplot(1, 2, 2)
#plt.imshow(img, cmap='gray', vmin=0, vmax=255)
#plt.show()

ent = entropy(img, variances)
huff_dict = huffmandict(huff_dict)
encoded_img = huffmanenco(img, huff_dict)
print(ent)
