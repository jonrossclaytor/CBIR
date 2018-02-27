#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(img):
    # convert to grayscale
    gray_img = to_gray(img)

    # generate keypoints and descriptions
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def showSIFTFeatures(img):
    color_img = img
    gray_img = to_gray(color_img)
    
    img_kp, img_desc = gen_sift_features(color_img)
    show_sift_features(gray_img, color_img, img_kp)