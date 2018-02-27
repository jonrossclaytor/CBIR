#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage(img):
    '''Function that accepts that path for an image and displays it'''
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))