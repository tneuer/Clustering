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
with open("Datafile3D.csv", "w") as f:
    np.random.seed(10)
    #multivariate norma clusters
    centers = [[-6,14, 1], [-20,-3, 8], [14,12, 3], [35,-2, 0]]
    cov = [[10,0.8,0.9], [1.5,8,-0.9], [4, -0.8, 13]]
    n = 3000
    positions = []
    for center in centers:
        for i in range(n):
            positions.append(stats.multivariate_normal.rvs(center, cov, 1))
    
    #connection of two clusters        
    connectionx = np.linspace(-19, -6, 15)
    connectiony = np.linspace(-3, 13, 15)
    connectionz = np.linspace(1, 8, 15)
    for i in range(len(connectionx)):
        positions.append((connectionx[i], connectiony[i], connectionz[i]))
    
    #uniformly distirbuted "background"
    b = 700
    xmin = min([c[0] for c in centers]) - 10
    xmax = max([c[1] for c in centers]) + 15
    
    for i in range(b):
        positions.append(np.random.uniform(xmin,xmax, size = (1,3))[0])
    csv_writer = csv.writer(f)
    csv_writer.writerows(positions)
  
if len(positions[0]) == 2:  
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    
    plt.scatter(x,y)
    
if len(positions[0]) == 3:
    fig = plt.figure()
    ax = Axes3D(fig)

    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[1] for pos in positions]
    
    for i in range(len(x)):
        ax.scatter(x[i],y[i],z[i], c="r")