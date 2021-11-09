import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import log2
from func import count_p, histogram, ref
import sys
# sys.path is a list of absolute path strings
sys.path.append('/Users/vadim.turko/Documents/GitHub/sommed/lab3')


img_name = 'test_images/kodim14.png'
img = cv.cvtColor(cv.imread(img_name)
                  [..., ::-1][:256, :256], cv.COLOR_BGR2GRAY)

p = count_p(img)
hist = histogram(img)


def entropy(img):
    ent = 0
    for _, val in np.ndenumerate(img):
        ent += p[val] * log2(p[val])
    return -ent


plt.subplot(1, 2, 1)
plt.bar(ref,  hist)
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

ent = entropy(img)
print(ent)
