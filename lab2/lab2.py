import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def rotate_image(image, angle):
  height = image.shape[0]
  width = image.shape[1]
  output=np.zeros((height, width, image.shape[2]))
  center_height = round(((image.shape[0]+1)/2)-1)
  center_width = round(((image.shape[1]+1)/2)-1) 

  angle=math.radians(angle)
  cosine=math.cos(angle)
  sine=math.sin(angle)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        y=image.shape[0]-1-i-center_height                  
        x=image.shape[1]-1-j-center_width                    

        new_y=round(-x*sine+y*cosine)
        new_x=round(x*cosine+y*sine)

        new_y=center_height-new_y
        new_x=center_width-new_x

        if 0 <= new_x < width and 0 <= new_y < height and new_x>=0 and new_y>=0:
            output[new_y,new_x,:]=image[i,j,:]
            
  normalized = cv.normalize(output.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
  return normalized

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    a = im[ y0, x0 ]
    b = im[ y1, x0 ]
    c = im[ y0, x1 ]
    d = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*a + wb*b + wc*c + wd*d

save_dir = 'result_images/'
name = 'kodim14'

img = cv.imread('test_images/' + name + '.png')[...,::-1]
print(img.shape)
rotated = rotate_image(img, 45)
scaled = bilinear_interpolate(img, 100, 100)
print(scaled)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(rotated)

plt.show()
