# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:08:34 2020

@author: keush
"""

import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2hsv
from skimage.color import yuv2rgb
from skimage import util

im = io.imread('ressources/images/poissons.jpg')
plt.imshow(im,cmap=plt.cm.gray,vmin=30, vmax=200)
plt.show()

inverted_img = util.invert(im)
plt.imshow(inverted_img,cmap=plt.cm.gray,vmin=30, vmax=200)
plt.show()

hsv_img = rgb2hsv(im)
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))

ax0.imshow(im)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')

fig.tight_layout()

yuv_img=yuv2rgb(im)
value_img=yuv_img[:,:,2]
plt.imshow(value_img)


