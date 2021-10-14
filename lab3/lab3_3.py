import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import func

img = func.get_image('3_1')

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

filtered_img = func.my_2dfilter(img, 15)
plt.subplot(1, 3, 2)
plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)

filtered_img = func.threshold_filter(filtered_img, 64)
plt.subplot(1, 3, 3)
plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)

plt.show()
