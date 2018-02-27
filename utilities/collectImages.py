import os
import numpy as np

from dbhelper import DBHelper 
from utilities.hashImage import hashImage
from utilities.readImage import readImage
from utilities.getOpening import getOpening

from colordescriptor import ColorDescriptor 

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

DB = DBHelper()

def collectImages(directory):
    '''Function that accepts a full path to a directory and adds all images found to the database'''

    # create schema (if necessary)
    DB.createSchema()
    
    i = 0
    images = []
    for fn in os.listdir(directory):
        full_path = directory + '\\' + fn

        filename, file_extension = os.path.splitext(full_path)
        img_arr = readImage(full_path)
        if type(img_arr) == np.ndarray:
            opened_image = getOpening(img_arr)

            img_hash = hashImage(full_path)
            color_vector = np.asarray(cd.describe(img_arr))
            image = [img_hash, full_path, img_arr, opened_image, color_vector]
            images.append(image)

            # commit every 100 images
            if i == 100:
            	DB.addImages(images)
            	i = 0
            	images = []
            else:
            	i += 1
            	
    # commit any remaining images
    if len(images) > 0:
    	DB.addImages(images)


    print 'Images added to database'