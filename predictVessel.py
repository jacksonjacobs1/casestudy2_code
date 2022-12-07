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

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, random_state=100)
    predVessels = km.fit_predict(testFeatures)

    #Takes ship speed and angle to return x,y component of ship's movement vector
    vector = testFeatures[:, [3,4]]
    x, y = vectorize(vector[:, 0], vector[:, 1])

    #Remove time, speed, angle as features and add the movement vector as features
    testFeatures = testFeatures[:, [1,2]]
    testFeatures = np.insert(testFeatures, 2, x, axis=1)
    testFeatures = np.insert(testFeatures, 3, y, axis=1)

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    batchSize = 1000
    bat = make_batch(testFeatures, batchSize)

    model = AgglomerativeClustering(n_clusters=numVessels, linkage='single')

    vid = np.zeros((batchSize ,int(len(testFeatures) / batchSize)))

    for i in range(0, bat.shape[2]):
        vid[:,i] = model.fit_predict(bat[:,:,i])
        if i != 0 :

            # last -i-1 to firt of i+1
            uniqueOldVid = np.unique(vid[:,i-1], return_index=True, return_inverse=True)
            firstIndexOld = uniqueOldVid[1]

            lastIndexOld = firstIndexOld
            for j in range(0, batchSize) :
                lastIndexOld[int(vid[j,i])] = j

            uniqueNewVid = np.unique(vid[:, i], return_index=True, return_inverse=True)
            firstIndexNew = uniqueNewVid[1]

            rowsOld = bat[0:len(lastIndexOld) ,:,i-1]
            for j in range(len(lastIndexOld)) :
                #vidOldIndex = vid[uniqueOldVid[2][j], i-1]
                rowsOld[j] = bat[uniqueOldVid[2][j], :, i-1]

            rowsNew = bat[0:len(firstIndexNew), :, i]
            for j in range(len(firstIndexNew)):
                #vidNewIndex = vid[uniqueNewVid[2][j], i]
                rowsNew[j] = bat[uniqueNewVid[2][j], :, i]

            for j in range(len(lastIndexOld)):
                #vidOldIndex = vid[j, i - 1]
                #rowsOld[j] = bat[j, :, i - 1]

                dfOld = pd.DataFrame(rowsOld)
                dfNew = pd.DataFrame(rowsNew)
                diff_df = dfNew - rowsOld[j]
                norm_df = diff_df.apply(np.linalg.norm, axis=1)
                selectedRow = dfNew.loc[norm_df.idxmin()]
                print(selectedRow.shape)
                selectedIndex = dfNew.index[[selectedRow]]

                selectedVid = vid[uniqueOldVid[2][j], i-1]

                vid[:,i] = np.where(vid[:,i] == selectedVID, selectedVid+1000, vid[:,i])

            # for j in uniqueVid :
            #     df = pd.DataFrame(bat[:,:,i])
            #     df[(df)]
            #     numUnique[j] = bat[]
            # numUnique = np.where(np.unique(vid[:,i])
    predVessels = model.fit_predict(testFeatures)
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 10, trainFeatures, trainLabels)

# given the Speed in knots and angle in Angles(thousands), convert to vector with x, y component
def vectorize(speed, angle) :
    x = np.multiply(speed, np.cos(np.radians(angle/10)))
    y = np.multiply(speed, np.sin(np.radians(angle/10)))
    return x, y

def make_batch(testFeatures, batchSize) :
    batches = np.zeros((batchSize, testFeatures.shape[1], int(len(testFeatures)/batchSize)))
    for index in range(0, int(len(testFeatures)/batchSize)):
        df = pd.DataFrame(testFeatures)
        df = df.iloc[index*batchSize : index*batchSize + batchSize]
        batches[:, :, index] = df.to_numpy()
    return batches

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    