import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import func

img = func.get_image('2_1')

area_sizes = [3, 5]

plt.subplot(1, len(area_sizes) + 1, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
for area_size in area_sizes:
    result_img = func.equalize_local(img, area_size)

    plt.subplot(1, len(area_sizes) + 1, 2 + area_sizes.index(area_size))
    plt.imshow(result_img, cmap='gray', vmin=0, vmax=255)

plt.show()
