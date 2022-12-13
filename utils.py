# -*- coding: utf-8 -*-
"""
Utility functions for working with AIS data

@author: Kevin S. Xu
"""

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import markers, colors
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def convertTimeToSec(timeVec):
    # Convert time from hh:mm:ss string to number of seconds
    return sum([a * b for a, b in zip(
        map(int, timeVec.decode('utf-8').split(':')), [3600, 60, 1])])


def loadData(filename):
    # Load data from CSV file into numPy array, converting times to seconds
    timestampInd = 2

    data = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1,
                      converters={timestampInd: convertTimeToSec})

    return data


def plotVesselTracks(latLon, clu=None, fig=None):
    # Plot vessel tracks using different colors and markers with vessels
    # given by clu

    n = latLon.shape[0]
    if clu is None:
        clu = np.ones(n)
    cluUnique = np.array(np.unique(clu), dtype=int)

    # plt.figure()
    if (fig == None):
        plt.figure()
    fig

    markerList = list(markers.MarkerStyle.markers.keys())

    normClu = colors.Normalize(np.min(cluUnique), np.max(cluUnique))
    for iClu in cluUnique:
        objLabel = np.where(clu == iClu)
        imClu = plt.scatter(
            latLon[objLabel, 0].ravel(), latLon[objLabel, 1].ravel(),
            marker=markerList[(iClu+1) % len(markerList)],
            c=clu[objLabel], norm=normClu, label=iClu)
        # plt.xlim([-76.34, -75.96])
        # plt.ylim([36.90, 37.052])

    if (fig == None):
        plt.colorbar(imClu)
        plt.legend().set_draggable(True)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

def reduce_classes_KNN(features:ndarray, predictions, num_labels):
	predictions_copy = [*predictions]
	unique,counts = np.unique(predictions_copy, return_counts=True)
	df = pd.DataFrame({'unique': unique, 'counts': counts})
	df = df.sort_values(by=['counts'], ascending=False).reset_index(drop=True)

	major_prediction_types = df.iloc[:num_labels]
	minor_predictions_types = df.iloc[num_labels:]

	print(predictions_copy)
	print(minor_predictions_types)

	major_idxs = []	# use to map indices of subspace to overall feature space indices
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