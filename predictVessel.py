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
    testFeatures = testFeatures[:, [1,2]]
    testFeatures = np.insert(testFeatures, 2, x, axis=1)
    testFeatures = np.insert(testFeatures, 3, y, axis=1)

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    from sklearn.manifold import TSNE
    scaledDownTestFeatures = TSNE(n_components=2, init='pca', early_exaggeration=30.0, perplexity=30.0,
                                  learning_rate='auto', n_jobs=-1).fit_transform(testFeatures)

    # km = KMeans(n_clusters=numVessels, random_state=100)
    from sklearn.cluster import DBSCAN
    #model = DBSCAN()
    model = AgglomerativeClustering(n_clusters=numVessels)
   #  model = KMeans(n_clusters=numVessels, random_state=100)
    predVessels = model.fit_predict(testFeatures)


    # # Prerequisite : BatchSize > numVessels
    # batchSize = 100
    # bat = make_batch(testFeatures, batchSize)
    #
    # model = AgglomerativeClustering(n_clusters=numVessels, linkage='single')
    #
    # vid = np.zeros((batchSize ,int(len(testFeatures) / batchSize)))
    #
    # # Run for the number of batches available
    # # Note: last batch may be smaller in size as compared to other batches
    # for i in range(0, bat.shape[2]):
    #     # predit the clusters in i'th batch
    #     if i == bat.shape[2]:
    #         vid[:,i] = model.fit_predict(testFeatures)
    #     vid[:,i] = model.fit_predict(bat[:,:,i])
    #     vid = mergePreviousClusters(vid, i, model, bat, numVessels)
    #
    #
    # predVessels = []
    # # Merge all the batch results into single output : predVessels
    # for i in range(0, bat.shape[2]):
    #     predVessels = np.append(predVessels, vid[:,i])
    #
    # predVessels = np.append(predVessels, vid[0:len(testFeatures) - int(len(testFeatures) / batchSize) * batchSize, 0])

    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Takes ship speed and angle to return x,y component of ship's movement vector
    vector = testFeatures[:, [3, 4]]
    x, y = vectorize(vector[:, 0], vector[:, 1])

    # Remove time, speed, angle as features and add the movement vector as features
    # testFeatures = testFeatures[:, [0, 1, 2]]
    testFeatures = np.insert(testFeatures, 5, x, axis=1)
    testFeatures = np.insert(testFeatures, 6, y, axis=1)

    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    from sklearn.manifold import TSNE
    scaledDownTestFeatures = TSNE(n_components=2, init='pca', early_exaggeration=30.0, perplexity=30.0, learning_rate='auto', n_jobs=-1).fit_transform(testFeatures)
    # Unsupervised prediction, so training data is unused
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=0.7, n_jobs=-1)
    predVessels = model.fit_predict(testFeatures)
    return predVessels

    # # Arbitrarily assume 20 vessels
    # return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

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
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN
    #combModel = KNeighborsClassifier(n_neighbors=1)
    #combModel = SVC()
    #combModel = LogisticRegression(solver='newton-cg')
    #combModel = DecisionTreeClassifier(max_leaf_nodes=numVessels)
    #combModel = RandomForestClassifier()
    combModel = DBSCAN()

    combModel.fit(featuresOfCurrentBatch, vidOfCurrentBatch[0])
    #combModel.fit(bat[:,:,i], vid[:,i])
    newClusterAssignments = combModel.predict(featuresOfPreviousBatch)

    print(newClusterAssignments)

    new2 = np.zeros(len(vid[:, i - 1])) - 1
    for j in range(0, len(featuresOfPreviousBatch)):
        new2 = np.where(vid[:,i-1] == vid[indexLastVidOfPreviousBatch[j],i-1], newClusterAssignments[j], new2)

    #newClusterAssignments = model.fit_predict(np.vstack((featuresOfCurrentBatch, featuresOfPreviousBatch)))

    # newClusterAssignments = []
    # selectedIndex = []
    # selectedRow = []
    # correspondingNewIndex = []
    # for j in range(0, len(featuresOfCurrentBatch)):
    #     dfOld = pd.DataFrame(featuresOfPreviousBatch)
    #     dfNew = pd.DataFrame(featuresOfCurrentBatch)
    #     diff_df = dfOld - dfNew[j]
    #     norm_df = diff_df.apply(np.linalg.norm, axis=1)
    #     selectedRow[j] = dfOld.loc[norm_df.idxmin()]
    #     minDistance[j] = norm_df[norm_df.idxmin()]
    #     selectedIndex[j] = dfOld.index[[selectedRow]]
    #     correspondingNewIndex[j] = j
    #     # newClusterAssignments[selectedIndex] = dfNew[j]
    #     # dfOld = dfOld.drop(selectedIndex)
    #
    # dfFeatures = pd.DataFrame({'col1':selectedRow, 'col2':selectedIndex, 'col3':minDistance, 'col4':correspondingNewIndex})
    # dfFeatures = dfFeatures.sort_values(by='col3')
    #
    # for j in range(0, len(featuresOfCurrentBatch)):
    #     rowIndex = dfFeatures[dfFeatures['col4']==j,2]
    #     new2 = np.where(vid[:,i-1] == vid[rowIndex ,2],i-1], , new2)


    # initialize the values to -1 then fill in the correct values
    # new = np.zeros(len(vid[:,i])) - 1
    # new2 = np.zeros(len(vid[:,i-1])) - 1
    #
    # for j in range(0, len(featuresOfCurrentBatch)):
    #     new = np.where(vid[:,i] == vid[indexFirstVidOfCurrentBatch[j],i], newClusterAssignments[j], new)
    # for j in range(0, len(featuresOfPreviousBatch)):
    #     new2 = np.where(vid[:,i-1] == vid[indexLastVidOfPreviousBatch[j],i-1], newClusterAssignments[j+len(featuresOfCurrentBatch)], new2)

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
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    numVessels = 10
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)

    predNumVessels = np.unique(predVesselsWithoutK).size

    output = np.unique(predVesselsWithoutK, return_counts=True)
    outputDf = pd.DataFrame(output[0], output[1])
    print(outputDf)
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
    plt.show()

# comments
#       for i in range(0, bat.shape[2]):
#     #     vid[:,i] = model.fit_predict(bat[:,:,i])
#     #     print(npvid[:,i].shape)
#     #     if i == 0 :
#     #         vid[:,i] = vid[:,i]+numVessels
#     #     if i != 0 :
#     #         # last -i-1 to firt of i+1
#     #         uniqueOldVid = np.unique(vid[:,i-1], return_index=True, return_inverse=True)
#     #         #print(len(uniqueOldVid[0]))
#     #         firstIndexOld = uniqueOldVid[1]
#     #
#     #         print(len(firstIndexOld))
#     #         lastIndexOld = firstIndexOld
#     #         for j in range(0, batchSize) :
#     #             lastIndexOld[int(vid[j,i])] = j
#     #
#     #         uniqueNewVid = np.unique(vid[:, i], return_index=True, return_inverse=True)
#     #         firstIndexNew = uniqueNewVid[1]
#     #
#     #         rowsOld = bat[0:len(lastIndexOld) ,:,i-1]
#     #         for j in range(len(lastIndexOld)) :
#     #             #vidOldIndex = vid[uniqueOldVid[2][j], i-1]
#     #             rowsOld[j] = bat[uniqueOldVid[2][j], :, i-1]
#     #
#     #         rowsNew = bat[0:len(firstIndexNew), :, i]
#     #         for j in range(len(firstIndexNew)):
#     #             #vidNewIndex = vid[uniqueNewVid[2][j], i]
#     #             rowsNew[j] = bat[uniqueNewVid[2][j], :, i]
#     #
#     #         #matchBatches = AgglomerativeClustering(n_clusters=numVessels, linkage='single')
#     #         finalVids = model.fit_predict(np.vstack((rowsOld, rowsNew)))
#     #         #print(len(np.vstack((rowsOld, rowsNew))))
#     #         finalVids = finalVids+i*numVessels
#     #         #print(finalVids)
#     #
#     #         # update the existing vids to reflect the clustered grouping
#     #         for j in range(0, batchSize):
#     #             for k in range(0, numVessels):
#     #                 # if (vid[j, i - 1] == uniqueOldVid[0][k] & i==1):
#     #                 #     vid[j, i - 1] = finalVids[k]
#     #                 # if (vid[j, i] == uniqueNewVid[0][k] & i==1):
#     #                 #     vid[j, i] = finalVids[k]
#     #                 if (vid[j, i] == uniqueNewVid[0][k]):
#     #                     vid[j, i] = finalVids[k]
#     #                 for l in range(1, i+1):
#     #                     if (vid[j, i - l] == uniqueOldVid[0][k]):
#     #                         vid[j, i - l] = finalVids[k]
#     #                     # if (vid[j, i] == uniqueNewVid[0][k]):
#     #                     #     vid[j, i] = finalVids[k]
#     #         print("i=", i, np.unique(vid[:, i]))
#     #         print("i-1 = ", i-1, np.unique(vid[:, i-1]))
#     #         if (i!= 1):
#     #             print("i-2 = ", i - 2, np.unique(vid[:, i - 2]))
#     #    # print(len(np.unique(vid[:,i])))
#     #
#     #
#     # predVessels = []
#     # for i in range(0, bat.shape[2]):
#     #     predVessels = np.append(predVessels, vid[:,i])
#     # #print(len(np.unique(predVessels)))
#     # predVessels = np.append(predVessels, vid[0:len(testFeatures)-int(len(testFeatures)/batchSize)*batchSize,0])
#
#
#             # for j in range(len(lastIndexOld)):
#             #     #vidOldIndex = vid[j, i - 1]
#             #     #rowsOld[j] = bat[j, :, i - 1]
#             #
#             #     dfOld = pd.DataFrame(rowsOld)
#             #     dfNew = pd.DataFrame(rowsNew)
#             #     diff_df = dfNew - rowsOld[j]
#             #     norm_df = diff_df.apply(np.linalg.norm, axis=1)
#             #     selectedRow = dfNew.loc[norm_df.idxmin()]
#             #     print(selectedRow.shape)
#             #     selectedIndex = dfNew.index[[selectedRow]]
#             #
#             #     selectedVid = vid[uniqueOldVid[2][j], i-1]
#             #
#             #     vid[:,i] = np.where(vid[:,i] == selectedVID, selectedVid+1000, vid[:,i])
#
#             # for j in uniqueVid :
#             #     df = pd.DataFrame(bat[:,:,i])
#             #     df[(df)]
#             #     numUnique[j] = bat[]
#             # numUnique = np.where(np.unique(vid[:,i])
#
    # newClusterAssignments = []
    # for j in range(0, len(featuresOfCurrentBatch)):
    #     dfOld = pd.DataFrame(featuresOfPreviousBatch)
    #     dfNew = pd.DataFrame(featuresOfCurrentBatch)
    #     diff_df = dfOld - dfNew[j]
    #     norm_df = diff_df.apply(np.linalg.norm, axis=1)
    #     selectedRow = dfOld.loc[norm_df.idxmin()]
    #     selectedIndex = dfOld.index[[selectedRow]]
    #     newClusterAssignments[selectedIndex] = dfNew[j]
    #     dfOld = dfOld.drop(selectedIndex)
    #predVessels = model.fit_predict(testFeatures)