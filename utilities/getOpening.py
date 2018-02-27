import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

def getOpening(img):
    '''Function that accepts an image array and returns the "opened" array
    https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    '''
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening