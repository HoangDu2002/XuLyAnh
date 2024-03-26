
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
img = cv2.imread("/content/hinh1.jpg", 0)

# binarize the image
binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
# binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
# binr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

#define the kernel
kernel = np.ones((7,7), np.uint8)

# invert the image
invert = cv2.bitwise_not(binr)

# erode the image
erosion = cv2.erode(invert, kernel, iterations=1)

# print the output
plt.subplot(1,3,1)
plt.title("Oragion Image"), plt.xticks([]),plt.yticks([])
plt.imshow(binr, cmap='gray')
plt.subplot(1,3,2)
plt.title("Invert Image"), plt.xticks([]),plt.yticks([])
plt.imshow(invert, cmap='gray')
plt.subplot(1,3,3)
plt.title("Erode Image"), plt.xticks([]),plt.yticks([])
plt.imshow(erosion, cmap='gray')





# Dilation the image
dilation = cv2.dilate(invert, kernel, iterations = 1)
# print the output
plt.subplot(1,3,1)
plt.title()



"""# Mục mới"""