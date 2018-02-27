import cv2
import dicom
import numpy as np


def readImage(path):
    '''Function that accepts a full path and returns the image array'''
    try:
        img = dicom.read_file(path)
        return 'dicom'
    except:
        try:
            img = cv2.imread(path)
        except:
            pass
    
    #img = img.astype(np.int64)
    return img