import cv2
from tkinter import Tk, Label, Button, filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

image = None  

def open_image():
    global image  
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.jfif")])
    if file_path:
        print("Open image function called")
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        desired_size = (350, 350)
        image = cv2.resize(image, desired_size)
        img_tk = ImageTk.PhotoImage(Image.fromarray(image))
        img_label.configure(image=img_tk)
        img_label.image = img_tk

def apply_filter():
    global image  

    selected_filter = filter_combobox.get()

    print("Apply filter function called. Selected filter:", selected_filter)

    if selected_filter == "SIFT":     
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        sift_image = cv2.drawKeypoints(gray, keypoints, image)
        filtered_image = cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB)

    elif selected_filter == "HOG":
        resized_img = cv2.resize(image, (64 * 4, 64 * 4))
        gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        filtered_image = hog_image

    elif selected_filter == "Phân đoạn ảnh K-Means":
        twoDimage = image.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        attempts=10
        ret, label, center = cv2.kmeans (twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS) 
        center = np. uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((image.shape))
        filtered_image = np.vstack((image, result_image))

    elif selected_filter == "Phân đoạn hình ảnh bằng phát hiện đường viền":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        __, thresh = cv2.threshold (gray, np. mean (gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate (cv2.Canny(thresh, 0, 255), None)
        cnt = sorted(cv2.findContours (edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) [-2], key=cv2.contourArea) [-1] 
        mask = np.zeros((300,500), np.uint8)
        masked = cv2.drawContours (mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(image, image, mask-mask) 
        segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        filtered_image = np.vstack((image, segmented))

    elif selected_filter == "Phân vùng hình ảnh bằng Ngưỡng Otsu":
        img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        thresh = threshold_otsu(img_gray)
        img_otsu = img_gray < thresh
        def filter (image, mask):
            r = image[:,:,0] * mask
            g = image[:,:,1] * mask
            b = image[:,:,2] * mask
            return np.dstack([r,g,b])
        filtered = filter (image, img_otsu)
        filtered_image = np.vstack((image, filtered))

    elif selected_filter == "Phân đoạn hình ảnh bằng cách sử dụng Mặt nạ màu":
        low = np.array([0, 0, 0])
        high = np.array([200, 170, 170])
        mask= cv2.inRange(image, low, high)
        result = cv2.bitwise_and (image, image, mask-mask)
        filtered_image = np.vstack((image, result))

    elif selected_filter == "Invert Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        invert = cv2.bitwise_not(binr)

        expanded_invert = np.expand_dims(invert, axis=2)
        expanded_invert = np.repeat(expanded_invert, 3, axis=2)

        filtered_image = np.vstack((image, expanded_invert))

    elif selected_filter == "Erosion Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        invert = cv2.bitwise_not(binr)

        erosion = cv2.erode(invert, kernel, iterations=1)
        
        expanded_erosion = np.expand_dims(erosion, axis=2)
        expanded_erosion = np.repeat(expanded_erosion, 3, axis=2)

        filtered_image = np.vstack((image, expanded_erosion))

    elif selected_filter == "Dilation Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        invert = cv2.bitwise_not(binr)

        dilation = cv2.dilate (invert, kernel, iterations=1)
        
        expanded_dilation = np.expand_dims(dilation, axis=2)
        expanded_dilation = np.repeat(expanded_dilation, 3, axis=2)

        filtered_image = np.vstack((image, expanded_dilation))

    elif selected_filter == "Opening Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        invert = cv2.bitwise_not(binr)
     
        opening = cv2.morphologyEx(invert, cv2.MORPH_OPEN, kernel, iterations = 1)
        
        expanded_opening = np.expand_dims(opening, axis=2)
        expanded_opening = np.repeat(expanded_opening, 3, axis=2)

        filtered_image = np.vstack((image, expanded_opening))

    elif selected_filter == "Closing Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        invert = cv2.bitwise_not(binr)
     
        closing = cv2.morphologyEx(invert, cv2.MORPH_CLOSE, kernel, iterations = 1)
        
        expanded_closing = np.expand_dims(closing, axis=2)
        expanded_closing = np.repeat(expanded_closing, 3, axis=2)

        filtered_image = np.vstack((image, expanded_closing))

    elif selected_filter == "Morphology Gradient Image":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        invert = cv2.bitwise_not(binr)
     
        morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel)
        
        expanded_morph_gradient = np.expand_dims(morph_gradient, axis=2)
        expanded_morph_gradient = np.repeat(expanded_morph_gradient, 3, axis=2)

        filtered_image = np.vstack((image, expanded_morph_gradient))

    elif selected_filter == "Edges Image":
        img_blur = cv2.GaussianBlur(image, (5,5), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(img_blur, 100, 200)

        expanded_edges = np.expand_dims(edges, axis=2)
        expanded_edges = np.repeat(expanded_edges, 3, axis=2)

        filtered_image = np.vstack((image, expanded_edges))

    elif selected_filter == "Sobel x Image":
        img_blur = cv2.GaussianBlur(image, (5,5), sigmaX=0, sigmaY=0)
        sobelx = cv2.Sobel(src = img_blur, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
        sobely = cv2.Sobel(src = img_blur, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
        sobelxy = cv2.Sobel(src = img_blur, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 5)

        sobelx_uint8 = cv2.convertScaleAbs(sobelx)
        sobelx_gray = cv2.cvtColor(sobelx_uint8, cv2.COLOR_BGR2GRAY)
        expanded_sobelx_gray = np.expand_dims(sobelx_gray, axis=2)
        expanded_sobelx_gray = np.repeat(expanded_sobelx_gray, 3, axis=2)

        sobely_uint8 = cv2.convertScaleAbs(sobely)
        sobely_gray = cv2.cvtColor(sobely_uint8, cv2.COLOR_BGR2GRAY)
        expanded_sobely_gray = np.expand_dims(sobely_gray, axis=2)
        expanded_sobely_gray = np.repeat(expanded_sobely_gray, 3, axis=2)

        sobelxy_uint8 = cv2.convertScaleAbs(sobelxy)
        sobelxy_gray = cv2.cvtColor(sobelxy_uint8, cv2.COLOR_BGR2GRAY)
        expanded_sobelxy_gray = np.expand_dims(sobelxy_gray, axis=2)
        expanded_sobelxy_gray = np.repeat(expanded_sobelxy_gray, 3, axis=2)

        filtered_image = np.vstack((image, expanded_sobelx_gray))

    elif selected_filter == "Sobel y Image":
        img_blur = cv2.GaussianBlur(image, (5,5), sigmaX=0, sigmaY=0)
        sobely = cv2.Sobel(src = img_blur, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = 5)

        sobely_uint8 = cv2.convertScaleAbs(sobely)
        sobely_gray = cv2.cvtColor(sobely_uint8, cv2.COLOR_BGR2GRAY)
        expanded_sobely_gray = np.expand_dims(sobely_gray, axis=2)
        expanded_sobely_gray = np.repeat(expanded_sobely_gray, 3, axis=2)

        filtered_image = np.vstack((image, expanded_sobely_gray))

    elif selected_filter == "Sobel xy Image":
        img_blur = cv2.GaussianBlur(image, (5,5), sigmaX=0, sigmaY=0)
        sobelxy = cv2.Sobel(src = img_blur, ddepth = cv2.CV_64F, dx = 1, dy = 1, ksize = 5)

        sobelxy_uint8 = cv2.convertScaleAbs(sobelxy)
        sobelxy_gray = cv2.cvtColor(sobelxy_uint8, cv2.COLOR_BGR2GRAY)
        expanded_sobelxy_gray = np.expand_dims(sobelxy_gray, axis=2)
        expanded_sobelxy_gray = np.repeat(expanded_sobelxy_gray, 3, axis=2)

        filtered_image = np.vstack((image, expanded_sobelxy_gray))

    elif selected_filter == "Chỉnh độ sáng và độ tương phản":
        brightness = 10
        # Adjusts the contrast by scaling the pixel values by 2.3
        contrast = 2.3 
        image2 = cv2.addWeighted (image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)

        brightness = 5
        contrast = 1.5
        image3 = cv2.addWeighted (image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)

        filtered_image = np.vstack((image, image3))

    elif selected_filter == "loại bỏ nhiễu khỏi ảnh":
        medianfilter_image = cv2.medianBlur(image, 3)
        Gaussian_image = cv2.GaussianBlur(image, (5,5), 0)

        filtered_image = np.vstack((medianfilter_image, Gaussian_image))

    elif selected_filter == "tăng cường màu sắc trong ảnh":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:,:,0] = image[:,:,0]*0.7
        image[:,:,1] = image[:,:,1]*1.5
        image[:,:,2] = image[:,:,2]*0.5
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        filtered_image = np.vstack((image, image2))

    elif selected_filter == "Cân bằng biểu đồ xám":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equal_histogram_image = cv2.equalizeHist(gray_image)
        filtered_image = np.vstack((equal_histogram_image))

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filtered_image = image

    img_tk = ImageTk.PhotoImage(Image.fromarray(filtered_image))
    img_label.configure(image=img_tk)
    img_label.image = img_tk


window = Tk()
window.geometry("850x500")
window.title("xla")


img_label = Label(window)
img_label.pack()


open_button = Button(window, text="Mo Anh", command=open_image)
open_button.place(x=10, y=10)


filter_combobox = ttk.Combobox(window, values=["None", "SIFT", "HOG",
                                               "Phân đoạn ảnh K-Means", 
                                               "Phân đoạn hình ảnh bằng phát hiện đường viền", 
                                               "Phân vùng hình ảnh bằng Ngưỡng Otsu", 
                                               "Phân đoạn hình ảnh bằng cách sử dụng Mặt nạ màu",
                                               "Invert Image",
                                               "Erosion Image",
                                               "Dilation Image",
                                               "Opening Image",
                                               "Closing Image",
                                               "Morphology Gradient Image",
                                               "Edges Image",
                                               "Sobel x Image",
                                               "Sobel y Image",
                                               "Sobel xy Image",
                                               "Chỉnh độ sáng và độ tương phản",
                                               "loại bỏ nhiễu khỏi ảnh",
                                               "tăng cường màu sắc trong ảnh"], state="readonly")
filter_combobox.configure(width=80)
filter_combobox.place(x = 10, y = 50)

apply_filter_button = Button(window, text="Tai Hinh Len", command=apply_filter)
apply_filter_button.place(x = 700, y = 10)

window.mainloop()
