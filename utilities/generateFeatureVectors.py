import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.preprocessing import scale

from dbhelper import DBHelper
DB = DBHelper()

from utilities.generateSiftFeatures import generateSiftFeatures
from utilities.generateORBFeatures import generateORBFeatures

def generateFeatureVectors(method='sift', use_opening=False):

    # delete any features from the DB if they are present
    DB.deleteFeatures()

    # load the clustering algorithm
    kmeans = joblib.load('kmeans.pkl')

    image_chunks = DB.getImages(use_opening)

    image_list = []
    feature_vectors = []

    for images_df in image_chunks:
        for i in range(0,len(images_df)):
            img_hash = images_df['img_hash'][i]
            img_arr = images_df['img_arr'][i]
            

            if method == 'sift':
                img_kp, img_desc = generateSiftFeatures(img_arr)
            elif method == 'orb':
                img_kp, img_desc = generateORBFeatures(img_arr)

            if type(img_desc) == np.ndarray:
                # add the image label as the first column in the array
                num_features = img_desc.shape[0]
                images = np.asarray([img_hash] * num_features)
                features = img_desc

                # get cluster predictions and combine them with the images
                data = scale(features)
                predictions = kmeans.predict(data)
                combined = zip(images, predictions)

                # convert to DataFrame
                df = pd.DataFrame(combined,columns=['img_hash','cluster'])

                # count the instances of clusters found in the image
                feature_series = df['cluster'].value_counts()

                # determine the clusters that were identified in this image
                represented_clusters = list(df['cluster'].value_counts().index)

                # add a "zero" instance to the feature series for any unrepresented vector
                for c in range(0,kmeans.n_clusters):
                    if c not in represented_clusters: 
                        feature_series.set_value(c,0)

                # sort the series by index
                feature_series = feature_series.sort_index()

                # convert the feature vector to a numpy array
                feature_counts = np.asarray(feature_series)

                # add the label and the feature vectors to the larger lists
                image_list.append(img_hash)
                feature_vectors.append(feature_counts)

    # combine the two lists
    image_features = zip(image_list, feature_vectors)

    # add features to the database
    DB.addFeatureVectors(image_features, method)

    print 'Features added to database'