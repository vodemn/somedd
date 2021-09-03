import cv2 as cv
import matplotlib.pyplot as plt

save_dir = 'result_images/'
name = 'kodim01'
img = cv.imread('test_images/' + name + '.png')

print('size: ', img.size, '\nheight: ',
      img.shape[0], '\nwidth: ', img.shape[1], '\nchannels: ', img.shape[2])

plt.matshow(img)
plt.colorbar()
plt.show()
cv.imshow("original", img)
cv.waitKey(0)

normalized = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
cv.imwrite(save_dir + name + '_normalized.png', normalized)
plt.matshow(normalized)
plt.colorbar()
plt.show()
cv.imshow("normalized", normalized)
cv.waitKey(0)

gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite(save_dir + name + '_gray_image.png', gray_image)
plt.matshow(gray_image)
plt.colorbar()
plt.show()
cv.imshow("gray_image", gray_image)
cv.waitKey(0)
