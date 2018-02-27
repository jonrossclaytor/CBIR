import cv2
import numpy as np


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


def generateSiftFeatures(img):

    try:
        img_kp, img_desc = gen_sift_features(img)
        # convert to int8
        img_desc = img_desc.astype(np.int8)

    except:
        return 'error', 'error'
    return img_kp, img_desc