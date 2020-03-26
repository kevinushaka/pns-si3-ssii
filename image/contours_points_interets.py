# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:02:21 2020

@author: keush
"""

from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
import cv2

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

image = io.imread('ressources/chat.jpg')
grayscale = rgb2gray(image)
edges_sobel = sobel_each(grayscale)
edges_prewitt = filters.prewitt(grayscale)
edges_gaussian = filters.gaussian(grayscale)
edges_sobel_h=filters.sobel_h(grayscale)
edges_sobel_v=filters.sobel_v(grayscale)
#plt.imshow(edges_sobel)
#plt.show()

import numpy as np
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('ressources/chatSIFT.jpg',image)


