import numpy as np
import cv2

def generateORBFeatures(img):
    ''' Function that accepts an image and returns the ORB keypoints and feature vector'''
    
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    return kp, des