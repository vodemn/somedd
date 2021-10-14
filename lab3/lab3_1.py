import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import func

image_count = 4

for i in range(image_count):
    img = func.get_image('1_' + str(i + 1))

    hist = np.zeros(func.ref.size)
    for k, v in func.count_p(img).items():
        hist[k] = v

    plt.subplot(image_count, 4, 1 + image_count * i)
    plt.bar(func.ref,  hist)
    plt.subplot(image_count, 4, 2 + image_count * i)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(image_count, 4, 3 + image_count * i)
    plt.hist(img)
    plt.subplot(image_count, 4, 4 + image_count * i)
    plt.imshow(func.equalize(img), cmap='gray', vmin=0, vmax=255)

plt.show()
