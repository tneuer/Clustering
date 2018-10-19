# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:04:52 2017

@author: thomas
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#read in the data

#with open("Datafile6D.csv", "r") as f:
#    positions = []
#    csv_reader = csv.reader(f)
#    for line in csv_reader:
#        positions.append([float(line[j]) for j in range(len(line))]) 
        
start = time.clock()
with open("CMSample1M.txt", "r") as f:
    #maximum = 100000
    nr_particles = 1000000
    nr_lines = int(nr_particles)
    i = 0
    positions = [i for i in range(nr_lines)]
    while (i < nr_lines):
        positions[i] = [float(entry) for entry in f.readline().split()]
        if i%10000 == 0:
            print(i)
        i += 1
read_in_time = time.clock() - start
        
def KMeans(data, k = None, starts = 5, maxIter = 50, plotter = False):
    """""""""""""""""""""""""""""
    Input:  data    -   List of tuples with at least two coordinates (x,y), 
                        but possibly more (x,y,z,vx,vy,vz)
            k       -   number of desired clusters, if None: 1-10 clusters
                        Always returns clusters where k == k_max
            starts  -   How often the algorithm starts with random positions,
                        Only the best clusters are returned
            maxIter -   Maxmimum number of itrations
            plotter -   if True, plots the steps of the algorithm for the 
                        coordinates (x,y)
            
    Output: clusters -  List of clusters, that contain all the points.
            centers  -  List of the positions of the cluster centers
            iterations - tuple of (nr_clusters, needed iterations)
            
    """"""""""""""""""""""""""""" 
    
    dim = len(data[0])
    nr_points = len(data)
    dataset = [[pos[i] for pos in data] for i in range(dim)]
    minima = [min(dataset[i]) for i in range(dim) ]
    maxima = [max(dataset[i]) for i in range(dim)]
    
    npdata = np.array(data)
    
    if k == None:
        k = [i for i in range(1,20)]
        plotter = False
    elif isinstance(k, int):
        k = [k]
    elif isinstance(k, list):
        pass
    else:
        print("k has to be an integer or a list!")
        return
    
    #Determine spatial dimensions, including eventually velocity
    if plotter and (dim == 2 or dim == 4):
        plotter2D = True
    else:
        plotter2D = False
        
    if plotter and (dim == 3 or dim == 6):
        plotter3D = True
    else:
        plotter3D = False
     
    #simple colormap for 3*7=21 clusters
    colors = ["#00ffff", "#ff00ff", "#99ff33", "#3333ff", "#ffff99", "#993333", "#ccffcc", "r", "g", "c", "m", "b"] * max(k)
    
    #Prepare 2D plot
    if plotter2D:
        plt.figure()
        ax = plt.gca()
        ax.set_axis_bgcolor((0, 0, 0))
        plt.xlim(minima[0], maxima[0])
        plt.ylim(minima[1], maxima[1])
     
    #Activates if potter = True to show progress
    def update_plot2D(c, ks, clusterIter, clusterMax, Iter, IterMax):            
        col = 0
        
        for cluster in clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            plt.scatter(x, y, s = 10, c = "{}".format(colors[col]))
            col += 1
            
        plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s = 30, c = colors)
        plt.title("Cluster: {}/{}, Iter_Cluster: {}/{}, Iter: {}/{}".format(
                    c, ks, clusterIter, clusterMax, Iter, IterMax))
        
        plt.pause(0.8)
        plt.cla()
        ax = plt.gca()
        ax.set_axis_bgcolor((0, 0, 0))
        plt.xlim(minima[0], maxima[0])
        plt.ylim(minima[1], maxima[1])
        
    #Prepare 3D plot
    if plotter3D:
        fig = plt.figure()
        ax = Axes3D(fig)
        
    def update_plot3D(c, ks, clusterIter, clusterMax, Iter, IterMax):      
        col = 0

        for cluster in clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            z = [point[2] for point in cluster]
            ax.scatter(x, y, z, c = "{}".format(colors[col]), s = 10)
            col += 1
            
        ax.scatter(np.array(centers)[:,0], np.array(centers)[:,1], np.array(centers)[:,2], s = 30, c = colors[:len(clusters)])
        plt.title("Cluster: {}/{}, Iter_Cluster: {}/{}, Iter: {}/{}".format(
                    c, ks, clusterIter, clusterMax, Iter, IterMax))
            
        plt.pause(0.8)
        plt.cla()

    
    #Calculates the mean of a number of vectors
    def mean(points, center):
        """
        Calculates the new position of a cluster center. If no point is in the 
        cluster, the center stays at the same spot
        """
        l = len(points)
        
        if l != 0:
            means = sum(np.array(points))/l
            return means
        else:
            return center
            
    #variable that counts which cluster number currently is chosen
    count = 0
    
    #sum of all Cluster_totalDistance in order to find the absolute best config
    totalSum = float("inf")
    
    #list of totlSum from different cluster numbers to support decisionmaking for k
    AllSums = [0 for i in range(len(k))]
    
    for ki in k:
        
        #Create a matrix with all points for every cluster
        dataMatrix = np.array([npdata for i in range(ki)])
        
        for iter_nr in range(starts):
            
            np.random.seed(np.random.randint(int(4294967295)))
            centersxyz = [np.random.uniform(minima[i], maxima[i], ki) for i in range(dim)] #initialize x,y,z,.. coordinates
            centers = np.array([[center[i] for center in centersxyz] for i in range(ki)]) #initialize center positions
            tmp_centers = [[0 for i in range(dim)] for i in range(ki)] #temporarily saves old center positions for comparison
            
            #Keeps track of iterations per k
            iterations = [[ki, 0] for ki in k]
            
            #False if clusters have converged to stationary point
            new_position = True
            
            while new_position  and iterations[count][1] < maxIter:                    
                    
                #Initialize list of clusters   
                clusters = [[] for i in range(len(centers))]
                
                #Initialize list of "quality of cluster" numbers
                cluster_totalDistance = [0 for i in range(len(centers))]
                
                dist = [np.sum((dataMatrix[i] - centers[i])**2, axis = 1) for i in range(ki)]
                minDist = np.min(dist, axis = 0)
                indexClosestCluster = np.argmin(dist, axis = 0)
                
                for i in range(nr_points):
                    clusters[indexClosestCluster[i]].append(dataMatrix[0][i])
                    cluster_totalDistance[indexClosestCluster[i]] += minDist[i]                                 
                    
                #Update Plot
                if plotter2D:
                    update_plot2D(ki, k, iter_nr, starts, iterations[count][1], maxIter)
                    
                if plotter3D:
                    update_plot3D(ki, k, iter_nr, starts, iterations[count][1], maxIter)
                
                
                new_position = False
                for i in range(ki):
                    tmp_centers[i] = np.array(centers[i])
                    centers[i] = mean(clusters[i], centers[i])
                    if (centers[i] - tmp_centers[i]).any():
                        new_position = True
                
                
                #Update iteration number
                iterations[count][1] += 1
            
            print(iter_nr, ki, iterations[count][1])
            
            #Checking if new cluster state is better than the one before
            if sum(cluster_totalDistance) < totalSum:
                totalSum = sum(cluster_totalDistance)
                clusters_final = clusters
                centers_final = centers
                iterations_final = iterations
                AllSums[count] = totalSum
                
        count += 1
    
    #Converting lcusters back to lists
    clusters = [[list(point) for point in cluster] for cluster in clusters]
    
    #Plotting Diagnostics if k has multiple values
    if len(k) != 1:
        plt.figure()
        x = [i for i in k]
        plt.plot(x, AllSums)        
        
    return (clusters_final, centers_final, iterations_final, AllSums)
  
#Set procedure  
lengths = [5000, 10000, 25000, 50000, 75000, 100000, 250000]
k = [100]
nr_start = 8
mIter = 150
doc_name = "KMeansBetter.txt"

with open(doc_name, "w") as f:
    f.write("Lengths: {} \n ".format(lengths))
    
for datalength in lengths:
    start = time.clock()
    np.random.seed(3)
    np.random.shuffle(positions)
    data = [[pos[0], pos[1], pos[2]] for pos in positions[:datalength]]

    dim = len(data[0])
    
    end1 = time.clock() - start
    
    start = time.clock()
    
    clusters, centers, iterations, SUMS = KMeans(data, k = k, starts = nr_start, maxIter = mIter, plotter = False)
    print(len(clusters))

    cluster_points = []
    for cluster in clusters:
        cluster_points.append(len(cluster))
    end2 = time.clock() - start

    with open(doc_name, "a") as f:
        f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {} \n\n Param: [k: {}, #Starts: {}, Max Iterations: {}] \n\n #Clusters: {}, Points per cluster: {} \n\n Centers: {} \n\n Rating: {} \n Iterations: {}".format(datalength, end1, end2, k, nr_start, mIter, len(clusters), cluster_points, centers, SUMS, iterations))        


print("Plotting...")
colors = ["#00ffff", "#ff00ff", "#99ff33", "#3333ff", "#ffff99", "#993333", "#ccffcc", "r", "g", "c", "m", "b"] * max(k)

############################################################################
"""""""""
    3D Plot color
"""""""""
############################################################################

start = time.clock()
fig = plt.figure()
ax = Axes3D(fig) 
ax.set_axis_bgcolor((0, 0, 0))
col = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    z = [point[2] for point in cluster]
    ax.scatter(x, y, z, c = "{}".format(colors[col]), s = 10)
    col += 1
#plt.show()
print(time.clock() - start)
        

############################################################################
"""""""""
    2D Plot color
"""""""""
############################################################################

start = time.clock()
fig = plt.figure()
ax = plt.gca()
ax.set_axis_bgcolor((0, 0, 0))
col = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, s = 7, c = "{}".format(colors[col]))
    col += 1
    
#plt.show()
print(time.clock() - start)


############################################################################
"""""""""
    2D Plot circles
"""""""""
############################################################################

start = time.clock()
fig = plt.figure()
ax = plt.gca()
it = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    cen = (np.mean(x), np.mean(y))
    xdist = max(x - cen[0])
    ydist = max(y - cen[1])
    rad = np.sqrt(xdist**2 + ydist**2)
    circle = plt.Circle(cen, rad, color = "r", fill = False)
    plt.scatter(x, y, c = "k", s = 7)
    ax.add_artist(circle)
    it += 1
                
    
print(time.clock() - start)


plt.show()
