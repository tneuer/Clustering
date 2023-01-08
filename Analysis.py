# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:34:38 2017

@author: thomas neuer
"""

# Added another thing

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

############################################################################
"""""""""
    Complexity Analysis on Cold Dark Matter
"""""""""
############################################################################

nr_points1 = [5000, 10000, 25000, 50000, 75000, 100000, 250000, 400000, 500000, 600000, 750000, 900000, 1000000]
nr_points2 = [5000, 10000, 25000, 50000, 75000, 100000] 
nr_points3 = [5000, 10000, 25000, 50000, 75000, 100000, 250000]

time1 = [0.5952349999999997, 0.6734100000000005, 1.7049070000000004, 2.8815229999999996, 4.929391999999998, 6.2660610000000005, 16.627046, 27.947475999999995, 35.35272500000001, 43.776341999999985, 59.00653, 72.432334, 80.84827100000001]
time21 = [0.7991510000000002, 3.4320040000000005, 23.88861, 116.09280500000001, 342.016356, 670.1044579999999]
time22 = np.sqrt(np.array(time21))
time31 = [12.318706999999998, 35.701794, 135.068039, 404.11595700000004, 725.902475, 1574.7550820000001, 3190.699606]
time32 = np.sqrt(np.array(time31))

plt.figure()

plt.semilogx(nr_points1, time1, label = "Density peak finder")
plt.scatter(nr_points1, time1)

plt.plot(nr_points2, time21, label = "FOAF")
plt.scatter(nr_points2, time21)
plt.plot(nr_points2, time22, label = "SQRT(FOAF)")
plt.scatter(nr_points2, time22)

plt.plot(nr_points3, time31, label = "KMeans")
plt.scatter(nr_points3, time31)
plt.plot(nr_points3, time32, label = "SQRT(KMeans)")
plt.scatter(nr_points3, time32)

plt.grid()
plt.legend(loc = 2)

plt.show()