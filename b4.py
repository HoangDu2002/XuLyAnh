
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("/content/playground.jfif")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)

twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)
print(twoDimage)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts=10

ret, label, center = cv.kmeans (twoDimage, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS) 
center = np. uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))

plt.axis('off')
plt.imshow(result_image)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("/content/anh1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
#plt.subplot(1,2,1)
plt.imshow(img)
#plt.show()

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
__, thresh = cv.threshold (gray, np. mean (gray), 255, cv.THRESH_BINARY_INV) 
plt.imshow(thresh)

edges = cv.dilate (cv.Canny(thresh, 0, 255), None)
plt.axis('off') 
plt.imshow(edges)

cnt = sorted(cv.findContours (edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) [-2], key=cv.contourArea) [-1] 
mask = np.zeros((300,500), np.uint8)
masked = cv.drawContours (mask, [cnt],-1, 255, -1)
plt.axis('off')
plt.imshow(masked)

dst = cv.bitwise_and(img, img, mask-mask) 
segmented = cv.cvtColor(dst, cv.COLOR_BGR2RGB) 
plt.imshow(segmented)

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2

sample_image = cv2.imread('/content/anh1.jpg') 
img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB) 
img = cv2.resize(img, (400,300))

plt.axis('off') 
plt.imshow(img)

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
thresh = threshold_otsu(img_gray)
img_otsu = img_gray < thresh
plt.imshow(img_otsu)

def filter_image (image, mask):
  r = image[:,:,0] * mask
  g = image[:,:,1] * mask
  b = image[:,:,2] * mask
  return np.dstack([r,g,b])

filtered = filter_image (img, img_otsu)

plt.axis('off')
plt.imshow(filtered)

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2

sample_image = cv2.imread('/content/anh1.jpg') 
img= cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

plt.axis('off') 
plt.imshow(img)

low = np.array([0, 0, 0])
high = np.array([200, 170, 170])
mask= cv2.inRange(img, low, high) 
plt.axis('off')
plt.imshow(mask)

result = cv2.bitwise_and (img, img, mask-mask)
plt.axis('off')
plt.imshow(result)