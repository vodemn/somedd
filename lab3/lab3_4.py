import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import func

img = func.get_image('4_1')

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

filtered_img = func.my_2dfilter(img, 3)
plt.subplot(1, 3, 2)
plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)

filtered_img = func.median_filter(filtered_img, 3)
plt.subplot(1, 3, 3)
plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)

plt.show()
