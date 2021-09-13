import cv2 as cv
import matplotlib.pyplot as plt

save_dir = 'result_images/'
name = 'kodim14'

img = cv.imread('test_images/' + name + '.png')
normalized = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)[...,::-1]
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print('size: ', img.size, '\nheight: ',
      img.shape[0], '\nwidth: ', img.shape[1], '\nchannels: ', img.shape[2])

      
plt.imsave(save_dir + name + '_normalized.png', normalized)
plt.imsave(save_dir + name + '_gray_image.png', gray_image, cmap='gray')

plt.subplot(2, 3, 1)
plt.imshow(img[...,::-1])
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(normalized)
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.colorbar()

plt.show()