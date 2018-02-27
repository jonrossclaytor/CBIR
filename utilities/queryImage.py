#%matplotlib inline

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colordescriptor import ColorDescriptor 

from dbhelper import DBHelper
DB = DBHelper()

from sklearn.externals import joblib
from sklearn.preprocessing import scale

from utilities.generateSiftFeatures import generateSiftFeatures
from utilities.generateORBFeatures import generateORBFeatures

from utilities.readImage import readImage
from utilities.showImage import showImage
from utilities.getOpening import getOpening

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

def chi2_distance(feature_vector, features, eps = 1e-10):
    dists = []
    # compute the chi-squared distance
    for i in range(0,len(features)):
        try:
            d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(feature_vector, features[i])])
            dists.append(d)
        except:
            print features[i], type(features[i])
    # return the chi-squared distance
    return np.asarray(dists)



def queryImage(image, N=3, feature_modifier=1, color_modifier=1, plot=True, use_opening=False):
    '''Function that accepts an image array and returns the N closest matches'''

    # collect the method and whether opening is used
    method = DB.retrieveMethod()

    # perform opening on the image
    if use_opening==True:
    	img = getOpening(image)
    else:
    	img = image

    # load the clustering algorithm
    kmeans = joblib.load('kmeans.pkl') 

    # generate features for the image
    if method == 'sift':
        img_kp, img_desc = generateSiftFeatures(img)
    elif method == 'orb':
        img_kp, img_desc = generateORBFeatures(img)

    data = scale(img_desc)

    # pass the image through the kmeans model
    predictions = kmeans.predict(data)

    # convert results to DataFrame
    df = pd.DataFrame(predictions,columns=['cluster'])

    # count the instances of clusters found in the image
    feature_series = df['cluster'].value_counts()

    # determine the clusters that were identified in this image
    represented_clusters = list(df['cluster'].value_counts().index)

    # add a "zero" instance to the feature series for any unrepresented cluster
    for c in range(0,kmeans.n_clusters):
        if c not in represented_clusters:
            feature_series.set_value(c,0)

    # sort the series by index
    feature_series = feature_series.sort_index()

    # convert the feature vector to a numpy array 
    feature_vector = np.asarray(feature_series)

    # collect the color vector
    colors_image = cd.describe(img)

    # convert colors to numpy array
    colors_image = np.asarray(colors_image,dtype=np.float64)

    # normalize the histogram
    colors_image = cv2.normalize(colors_image,colors_image)

    # START MATRIX

    # collect library of images
    images, features, colors = DB.collectFeatures()

    # normalize the histogram
    colors = cv2.normalize(colors,colors)

    # calculate the chi squared distance between the feature vector and each image in the matrix
    dist_features = chi2_distance(feature_vector, features)

    # calculate the chi squared distance between the normalized color histograms
    dist_colors = chi2_distance(colors_image, colors)

    # combine the images and the distances and convert to DataFrame
    results = pd.DataFrame(zip(images, dist_features, dist_colors), columns=['img_hash','dist_features','dist_colors']).reset_index(drop=True)

    # add the ranks and combined scores
    results['dist_features_rank'] = results['dist_features'].rank()
    results['dist_colors_rank'] = results['dist_colors'].rank()

    results['overall_rank'] = ((results['dist_features_rank'] * feature_modifier) + (results['dist_colors_rank'] * color_modifier)) / (feature_modifier + color_modifier)

    results = results.sort_values('overall_rank', ascending=True)


    if plot == False:
        return list(results['img_hash'][:N])
    else:
        # collect top N images from the database
        matches = ''
        for img in list(results['img_hash'][:N]):
            matches += ",'" + img + "'"
        matches = matches[1:] # omit initial comma


        # retrieve matched images from the database
        img_matches = DB.retrieveMatches(matches)
        
        # plot
        for i in range(1,N+1):
            plt.subplot(1, N, i)
            showImage(img_matches[i-1])
    
