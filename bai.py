import numpy as np
import pandas as pd
import cv2 as cv
import requests
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
"""
image_urls = ['https://iiif.lib.ncsu.edu/iiif/0052574/full/800,/0/default.jpg',

       'https://iiif.lib.ncsu.edu/iiif/0016007/full/800,/0/default.jpg',

      'https://placekitten.com/800/571']  

for url in image_urls:
    img = Image.open(requests.get(url, stream=True).raw)
    img.show()
"""

#img = cv.imread('C:\\Users\\Admin\\Desktop\\Anh\\Ảnh 1.jpg', 1)
for file in glob.glob('C:\\Users\\Admin\\Desktop\\Anh\\*.jpg'):
    img = cv.imread(file)
    
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image)

brightness = 10

contrast = 2.3
image2 = cv.addWeighted(image, contrast, np.zeros(image.shape, image.dtype),0, brightness)

cv.imwrite('modified_image.jpg', image2)
plt.subplot(1, 3, 2)
plt.title("Brightness = 10 & contrast = 2.3")
plt.imshow(image2)
brightness = 5
contrast = 1.5
image3 = cv.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)

plt.subplot(1 , 3, 3)
plt.title("Brightness = 5 & contarst = 1.5")
plt.imshow(image3)

plt.show()