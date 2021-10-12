import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

img = plt.imread('1_1.tif')
print(img)

plt.subplot(1, 3, 1)
plt.imshow(cv.normalize(img.astype(
    'float'), None, 0.0, 1.0, cv.NORM_MINMAX))

plt.show()
