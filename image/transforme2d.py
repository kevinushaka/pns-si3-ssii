# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:08:34 2020

@author: keush
"""

import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate

im = io.imread('ressources/images/poissons.jpg')
im=rotate(im,15,resize=True,order=5)
plt.imshow(im)
plt.show()
