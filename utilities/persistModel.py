import numpy as np

from dbhelper import DBHelper
DB = DBHelper()

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from sklearn.externals import joblib

from utilities.generateSiftFeatures import generateSiftFeatures
from utilities.generateORBFeatures import generateORBFeatures
from utilities.generateFeatureVectors import generateFeatureVectors

def persistModel(method='sift', clusters=5, use_opening=False):
    # collect images from database
    image_chunks = DB.getImages(use_opening)

    # initialize the model
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=clusters, n_init=10, batch_size=100)

    for images_df in image_chunks:
        for i in range(0,len(images_df)):
            img_hash = images_df['img_hash'][i]
            img_arr = images_df['img_arr'][i]

            if method == 'sift':
                img_kp, img_desc = generateSiftFeatures(img_arr)
            elif method == 'orb':
                img_kp, img_desc = generateORBFeatures(img_arr)

            if type(img_desc) == np.ndarray:
                if i == 0:
                    features = img_desc
                else:
                    features = np.concatenate((features,img_desc),axis=0) # vertically

        data = scale(features)
        kmeans.partial_fit(data)  

    joblib.dump(kmeans, 'kmeans.pkl')

    print "model saved as 'kmeans.pkl'"

    # generate features with the model
    generateFeatureVectors(method=method, use_opening=use_opening)