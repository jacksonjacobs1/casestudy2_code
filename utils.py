# -*- coding: utf-8 -*-
"""
Utility functions for working with AIS data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers, colors


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
            marker=markerList[iClu % len(markerList)],
            c=clu[objLabel], norm=normClu, label=iClu)

    if (fig == None):
        plt.colorbar(imClu)
        plt.legend().set_draggable(True)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
