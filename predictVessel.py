# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused

    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    # km = KMeans(n_clusters=numVessels, random_state=100)
    # predVessels = km.fit_predict(testFeatures)

    # Takes ship speed and angle to return x,y component of ship's movement vector
    vector = testFeatures[:, [3,4]]
    x, y = vectorize(vector[:, 0], vector[:, 1])

    #Remove time, speed, angle as features and add the movement vector as features
    testFeatures = testFeatures[:, [0,1,2]]
    testFeatures = np.insert(testFeatures, 3, x, axis=1)
    testFeatures = np.insert(testFeatures, 4, y, axis=1)

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    from sklearn.manifold import TSNE
    # scaledDownTestFeatures = TSNE(n_components=3, init='pca',
    #                               learning_rate='auto', n_jobs=-1).fit_transform(testFeatures)

    # km = KMeans(n_clusters=numVessels, random_state=100)
    from sklearn.cluster import DBSCAN
    #model = DBSCAN()
    # model = AgglomerativeClustering(n_clusters=numVessels)
    model = KMeans(n_clusters=numVessels, random_state=100)
    predVessels = model.fit_predict(testFeatures)

    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Takes ship speed and angle to return x,y component of ship's movement vector
    vector = testFeatures[:, [3, 4]]
    x, y = vectorize(vector[:, 0], vector[:, 1])

    # Remove time, speed, angle as features and add the movement vector as features
    testFeatures = testFeatures[:, [0, 1, 2, 3, 4]]
    testFeatures = np.insert(testFeatures, 5, x, axis=1)
    testFeatures = np.insert(testFeatures, 6, y, axis=1)

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # testFeatures[:, [3]] = testFeatures[:, [3]]*2

    from sklearn.manifold import TSNE
    # scaledDownTestFeatures = TSNE(n_components=2, init='pca',learning_rate='auto', n_jobs=-1).fit_transform(testFeatures)
    # Unsupervised prediction, so training data is unused

    from sklearn.cluster import DBSCAN
    from sklearn.cluster import OPTICS



    if trainLabels is not None:
        score = np.zeros(10)
        params = np.arange(10) * 0.1 + 0.1
        for i in range(0, len(params)):
            model = DBSCAN(eps=params[i], n_jobs=-1)
            predVessels = model.fit_predict(testFeatures)
            score[i] = adjusted_rand_score(trainLabels, predVessels)
            print(i, ":", score[i])

        model = DBSCAN(eps=params[1], n_jobs=-1)
        predVessels = model.fit_predict(testFeatures)

    if trainLabels is None:
        # model = DBSCAN(eps=0.7, n_jobs=-1)
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=150, min_samples=10, cluster_selection_epsilon=0.5)
        # model = OPTICS(cluster_method='dbscan', eps=0.6, n_jobs=-1)
        predVessels = model.fit_predict(testFeatures)

        predVessels = reduce_classes_KNN(testFeatures, predVessels, 11)

    return predVessels

    # # Arbitrarily assume 20 vessels
    # return predictWithK(testFeatures, 20, trainFeatures, trainLabels)


def reduce_classes_KNN(features, predictions, num_labels):
    predictions_copy = [*predictions]
    unique, counts = np.unique(predictions_copy, return_counts=True)
    df = pd.DataFrame({'unique': unique, 'counts': counts})
    df = df.sort_values(by=['counts'], ascending=False).reset_index(drop=True)

    major_prediction_types = df.iloc[:num_labels]
    minor_predictions_types = df.iloc[num_labels:]

    print(predictions_copy)
    print(minor_predictions_types)

    major_idxs = []  # use to map indices of subspace to overall feature space indices
    minor_idxs = []

    major_feat_space = []
    minor_feat_space = []

    # separate feature space into two. One contains "minor class" points, and the other contains "major class" points.
    for idx, pred in enumerate(predictions_copy):
        if pred in minor_predictions_types['unique'].tolist():
            minor_idxs.append(idx)
            minor_feat_space.append(features[idx])
        else:
            major_idxs.append(idx)
            major_feat_space.append(features[idx])

    print(f'major feat space shape: {np.array(major_feat_space).shape}')
    print(f'minor feat space shape: {np.array(minor_feat_space).shape}')

    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(major_feat_space)

    _, nn_inds = knn.kneighbors(minor_feat_space)

    for minor_idx, item in enumerate(nn_inds):
        item = item[0]
        current_prediction = predictions_copy[minor_idxs[minor_idx]]
        closest_prediction = predictions_copy[major_idxs[item]]

        predictions_copy[minor_idxs[minor_idx]] = closest_prediction
        print(f'{current_prediction} {closest_prediction}')

    print(predictions_copy)
    return np.array(predictions_copy)

# given the Speed in knots and angle in Angles(thousands), convert to vector with x, y component
def vectorize(speed, angle) :
    x = np.multiply(speed, np.cos(np.radians(angle/10)))
    y = np.multiply(speed, np.sin(np.radians(angle/10)))
    return x, y

# Given the fentire eature space and batchsize, returns an array where each element contains "batchSize" features
def make_batch(testFeatures, batchSize) :
    batches = np.zeros((batchSize, testFeatures.shape[1], int(len(testFeatures)/batchSize)))
    for index in range(0, int(len(testFeatures)/batchSize)):
        df = pd.DataFrame(testFeatures)
        df = df.iloc[index*batchSize : index*batchSize + batchSize]
        batches[:, :, index] = df.to_numpy()
    return batches

# Given vid (array of matrices where each entry is a array of the predicted vessel cluster for that specific batch)
# and i, rename the cluster assignment to merge the clusters
def mergePreviousClusters(vid, i, model, bat, numVessels) :
    if i == 0:
        return vid

    vidOfCurrentBatch = np.unique(vid[:,i], return_index=True)
    #print(i, vidOfCurrentBatch[0])
    indexFirstVidOfCurrentBatch = np.unique(vid[:,i], return_index=True)[1]
    #vidOfPreviousBatch = np.unique(vid[:,i-1])
    # next line works if vid[:,i-1] is numpy array
    indexLastVidOfPreviousBatch = (len(vid[:, i - 1]) - 1) - np.unique(np.flip(vid[:, i - 1]), return_index=True)[1]


    featuresOfCurrentBatch = bat[indexFirstVidOfCurrentBatch,:,i]
    featuresOfPreviousBatch = bat[indexLastVidOfPreviousBatch,:,i-1]

    # fundamental problem here
    from sklearn.ensemble import RandomForestClassifier
    combModel = RandomForestClassifier()


    combModel.fit(featuresOfCurrentBatch, vidOfCurrentBatch[0])
    #combModel.fit(bat[:,:,i], vid[:,i])
    newClusterAssignments = combModel.predict(featuresOfPreviousBatch)

    new2 = np.zeros(len(vid[:, i - 1])) - 1
    for j in range(0, len(featuresOfPreviousBatch)):
        new2 = np.where(vid[:,i-1] == vid[indexLastVidOfPreviousBatch[j],i-1], newClusterAssignments[j], new2)

    #newClusterAssignments = model.fit_predict(np.vstack((featuresOfCurrentBatch, featuresOfPreviousBatch)))

    #vid[:,i] = new
    vid[:,i-1] = new2

    return mergePreviousClusters(vid, i-1, model, bat, numVessels)



# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set3noVID.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    # plotVesselTracks(features[:,[2,1]])
    # plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    numVessels = 10
    predVesselsWithK = predictWithK(features, numVessels)

    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    # predVesselsWithoutK = predictWithoutK(features, None, labels)

    predNumVessels = np.unique(predVesselsWithoutK).size

    output = np.unique(predVesselsWithoutK, return_counts=True)
    outputDf = pd.DataFrame(output[0], output[1])
    # print(outputDf)
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    # plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    # plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    # plotVesselTracks(features[:,[2,1]], labels)
    # plt.title('Vessel tracks by label')
    plt.show()

