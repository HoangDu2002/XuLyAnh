

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# reading the image
img = imread('/content/anh.jpg')
plt.axis ("off")
plt.imshow(img) 
print(img.shape)

# resizing image
resized_img = resize(img, (64*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
print (resized_img.shape)

#creating hog features
fd, hog_image = hog (resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")

import cv2
# reading the image
img = cv2.imread('/content/anh8.jpg')
# convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create SIFT feature extractor
sift = cv2.xfeatures2d. SIFT_create()
# detect features from the image
keypoints, descriptors = sift.detectAndCompute (img, None)
# draw the detected key points
sift_image = cv2.drawKeypoints (gray, keypoints, img)
# show the image
plt.axis("off")
plt.imshow(sift_image)
# save the image
cv2.imwrite("lena_sift_feature.jpg", sift_image)