import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def rotate_image(image, angle):
    h, w, c = image.shape
    output = np.zeros((h, w, c))
    center_height = round(h / 2)
    center_width = round(w / 2)

    angle = math.radians(angle)
    cosine = math.cos(angle)
    sine = math.sin(angle)

    for i in range(h):
        for j in range(w):
            y = h-1-i-center_height
            x = w-1-j-center_width

            new_y = round(-x*sine+y*cosine)
            new_x = round(x*cosine+y*sine)

            new_y = center_height-new_y
            new_x = center_width-new_x

            if 0 <= new_x < w and 0 <= new_y < h and new_x >= 0 and new_y >= 0:
                output[new_y, new_x, :] = image[i, j, :]

    return output


def enlarge2(image):
    h, w, c = image.shape
    new_x = w * 2 - 1
    new_y = h * 2 - 1
    image_y = np.zeros((new_y, w, c))
    for i in range(h - 1):
        image_y[i * 2, :, :] = image[i, :, :]
        image_y[i * 2 + 1, :, :] = (image[i, :, :] + image[i + 1, :, :]) / 2
    image_y[new_y - 1] = image[h - 1]

    result = np.zeros((new_y, new_x, c))
    for j in range(w - 1):
        result[:, j * 2, :] = image_y[:, j, :]
        result[:, j * 2 + 1, :] = (image_y[:, j, :] + image_y[:, j + 1, :]) / 2

    return result


save_dir = 'result_images/'
name = 'kodim14'

img = cv.imread('test_images/' + name + '.png')[..., ::-1]
normalized = cv.normalize(img.astype(
    'float'), None, 0.0, 1.0, cv.NORM_MINMAX)
rotated = rotate_image(normalized, 45)
doubled = enlarge2(normalized)


plt.imsave('result' + '_sclaed.png', doubled)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(rotated)
plt.subplot(1, 3, 3)
plt.imshow(doubled)

plt.show()
