# -*- coding: utf-8 -*-
"""
Script to evaluate prediction accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import pandas as pd
import time

from utils import loadData, plotVesselTracks
from predictVessel import predictWithK, predictWithoutK

#%% Load training and test data. Training data may not necessarily be used.
testData = loadData('set2.csv')
testFeatures = testData[:,2:]
testLabels = testData[:,1]
trainData = loadData('set1.csv')
trainFeatures = trainData[:,2:]
trainLabels = trainData[:,1]

#allData = trainData
#trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
    #allData[:,2:], allData[:,1], test_size=0.5)

#%% Run prediction algorithms and check accuracy
#numVessels = np.unique(allData[:,1]).size
numVessels = np.unique(testLabels).size
predVesselsWithK = predictWithK(testFeatures, numVessels, trainFeatures,
                                trainLabels)
# Check to ensure that there are at most K vessels. If not, set adjusted
# Rand index to -infinity to indicate an invalid result (0 accuracy score)
if np.unique(predVesselsWithK).size > numVessels:
    ariWithK = -np.inf
else:
    ariWithK = adjusted_rand_score(testLabels, predVesselsWithK)

predVesselsWithoutK = predictWithoutK(testFeatures, trainFeatures, trainLabels)
predNumVessels = np.unique(predVesselsWithoutK).size
ariWithoutK = adjusted_rand_score(testLabels, predVesselsWithoutK)

print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
      + f'{ariWithoutK}')

# # %% Plot vessel tracks colored by prediction and actual labels
plotVesselTracks(testFeatures[:,[2,1]], predVesselsWithK)
plt.title('Vessel tracks by cluster with K')
plotVesselTracks(testFeatures[:,[2,1]], predVesselsWithoutK)
plt.title('Vessel tracks by cluster without K')
plotVesselTracks(testFeatures[:,[2,1]], testLabels)
plt.title('Vessel tracks by label')
plt.show()




# df = pd.read_csv('set3noVID.csv')
# df.sort_values('SEQUENCE_DTTM')
# bucSize = 100
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
# plt.legend().set_draggable(True)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
#
# for index in range(0, len(df), bucSize) :
#     temp = df.iloc[index:index + bucSize]
#     temp2 = temp[['LAT', 'LON']]
#     temp3 = temp2.to_numpy()
#     #print(temp3.shape)
#     plotVesselTracks(temp3, None, fig)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.show()
#     time.sleep(0.2)
#     #plt.clf()





