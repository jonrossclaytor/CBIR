import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale

from dbhelper import DBHelper
DB = DBHelper()

from utilities.generateSiftFeatures import generateSiftFeatures
from utilities.generateORBFeatures import generateORBFeatures

def evaluateKMeans(cluster_list=[5,10,15], sample=1, use_opening=False):
    '''Function that evaluates clusters with different K values'''

    image_chunks = DB.getImages(use_opening) 

    # find the silhouette_score
    for c in cluster_list:
        concatenated_feature_array = 'empty'

        sift_estimator = None
        sift_estimator = MiniBatchKMeans(init='k-means++', n_clusters=c, n_init=10, batch_size=500)

        orb_estimator = None
        orb_estimator = MiniBatchKMeans(init='k-means++', n_clusters=c, n_init=10, batch_size=500)

        for images_df in image_chunks: 

            # take a random sample from the DataFrame
            images_df = images_df.sample(frac=sample)
            images_df = images_df.reset_index(drop=True)

            estimates = [] 

            for i in range(0,len(images_df)):
                # collect features for the images in the chunk
                img_hash = images_df['img_hash'][i]
                img_arr = images_df['img_arr'][i]

                sift_img_kp, sift_img_desc = generateSiftFeatures(img_arr)
                orb_img_kp, orb_img_desc = generateORBFeatures(img_arr)

                if type(sift_img_desc) == np.ndarray and type(orb_img_desc) == np.ndarray: # confirm that features could be taken from the image
                    if concatenated_feature_array == 'empty':
                        sift_features = sift_img_desc
                        orb_features = orb_img_desc
                        concatenated_feature_array = 'not empty'
                    else:
                        sift_features = np.concatenate((sift_features,sift_img_desc),axis=0) # vertically  
                        orb_features = np.concatenate((orb_features,orb_img_desc),axis=0) # vertically  

        # scale the data
        sift_data = scale(sift_features)
        orb_data = scale(orb_features)

        # fit the models and add the estimates
        sift_estimator.partial_fit(sift_data)
        row = ['sift', c, metrics.silhouette_score(sift_data, sift_estimator.labels_,metric='euclidean', sample_size=25000)]
        estimates.append(row)


        orb_estimator.partial_fit(orb_data)
        row = ['orb', c, metrics.silhouette_score(orb_data, orb_estimator.labels_,metric='euclidean', sample_size=25000)]
        estimates.append(row)


    # convert to DataFrame
    estimates_df = pd.DataFrame(estimates,columns=['method','clusters','silhouette_score'])


    # extract sections for plotting
    clusters_axis = estimates_df.groupby(['clusters'])['silhouette_score'].mean().index.values
    sift_df = estimates_df[estimates_df['method'] == 'sift']
    orb_df = estimates_df[estimates_df['method'] == 'orb']

    sift_silhouette = np.asarray(sift_df.groupby(['clusters'])['silhouette_score'].mean())
    orb_silhouette = np.asarray(orb_df.groupby(['clusters'])['silhouette_score'].mean())

    # plot
    fig, ax = plt.subplots()
    ax.plot(clusters_axis, sift_silhouette, 'b--', label='SIFT Featurues')
    ax.plot(clusters_axis, orb_silhouette, 'r--', label='ORB Featurues')

    legend = ax.legend(loc='best', shadow=False, fontsize='x-large')
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')

    legend.get_frame().set_facecolor('#FFFFFF')

    plt.show()