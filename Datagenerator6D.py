# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:01:58 2017

@author: thomas
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

##########Create random positions and save them in a csv file
with open("Datafile6D.csv", "w") as f:
    np.random.seed(10)
    #multivariate norma clusters
    centers = [[-45,43, 5, 12, -5,-7], [-12,-3, 58, 10, -4, -6], [-89,-120, -30, 11, -4, 12], [60,-19, 12, -11, -17, -15], [33, -1, 1, 8, 9,10], [150, -80, 77, 8, 9,10], [75, 114, 55, 8, 9,10]]
    cov = [[80,0.8,0.9,0.5,0.3,0.1], [1.5,96,-0.9,-0.2,0.1,0.4], [4, -0.8, 106,0.4,0.3,0.2], [4, -0.8, 13,0.4,0.3,0.2], [4, -0.8, 13,0.4,0.3,0.2], [4, -0.8, 13,0.4,0.3,0.2]]
    n = 30000
    positions = []
    for center in centers:
        for i in range(n):
            positions.append(stats.multivariate_normal.rvs(center, cov, 1))
    
    #connection of two clusters        
    connectionx = np.linspace(-45, -12, 100)
    connectiony = np.linspace(43, -3, 100)
    connectionz = np.linspace(5, 58, 100)
    connectionvx = np.linspace(10,12,100)
    connectionvy = np.linspace(-5,-4,100)
    connectionvz = np.linspace(-7,-6,100)
    for i in range(len(connectionx)):
        positions.append((connectionx[i], connectiony[i], connectionz[i], connectionvx[i], connectionvy[i], connectionvz[i]))
    
    #uniformly distirbuted "background"
    b = 30000
    xmin = min([c[0] for c in centers]) - 10
    xmax = max([c[1] for c in centers]) + 15
    
    for i in range(b):
        positions.append(np.random.uniform(xmin,xmax, size = (1,6))[0])
    csv_writer = csv.writer(f)
    csv_writer.writerows(positions)
    
#if len(positions[0]) == 3 or len(positions[0]) == 6:
#    fig = plt.figure()
#    ax = Axes3D(fig)
#
#    x = [pos[0] for pos in positions]
#    y = [pos[1] for pos in positions]
#    z = [pos[2] for pos in positions]
#    
#    for i in range(len(x)):
#        ax.scatter(x[i],y[i],z[i], c="r")