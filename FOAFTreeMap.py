# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:56:56 2017

@author: thomas
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import copy


#read in the data
with open("Datafile6D.csv", "r") as f:
    positions = []
    csv_reader = csv.reader(f)
    for line in csv_reader:
        positions.append([float(line[j]) for j in range(len(line))])   

def FOAF(data, dist, density = 0, metric = "euclid", splits = 0):
    """""""""""""""""""""""""""""
    Input:  data    -   List of tuples with at least two 
                        coordinates (x,y), but possibly more (x,y,z,vx,vy,vz)
            dist    -   friends radius for the metric
            density -   Density constraint on points
            metric  -   metric to determine the distance of two points
            plotter -   if True, plots the steps of the algorithm for the 
                        coordinates (x,y)
            
    Output: friends -   List of clusters, that contain all the points.
                        Cluster 0 are all the points without friends               
    """""""""""""""""""""""""""""
                
    cluster_nr = 1 #cluster number 0 for unclustered points
    d2 = dist**2 #distance squared makes operations easier
    dim = len(data[0])
    d2 = [d2 for i in range(200)]    
    
    s = time.clock()
    if splits != 0:
        if dim == 3 or dim == 6:
            neighbouring_boxes = [[-1,-1,-1], [-1,-1,0], [-1,-1,1], [-1,0,-1], [-1,0,0], [-1,0,1], [-1,1,-1], [-1,1,0], [-1,1,1], [0,-1,-1], [0,-1,0], [0,-1,1], [0,0,-1], [0,0,0], [0,0,1], [0,1,-1], [0,1,0], [0,1,1], [1,-1,-1], [1,-1,0], [1,-1,1], [1,0,-1], [1,0,0], [1,0,1], [1,1,-1], [1,1,0], [1,1,1]]
            KDTree = [[[[] for i in range(splits[2])] for j in range(splits[1])] for l in range(splits[0])]
        if dim == 2 or dim == 4:
            neighbouring_boxes = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
            KDTree = [[[] for j in range(splits[1])] for l in range(splits[0])]
        dataset = [[pos[i] for pos in data] for i in range(dim)]
        minima = [min(dataset[i]) for i in range(dim) ]
        maxima = [max(dataset[i]) for i in range(dim)]
        intervals = [(maxima[i]-minima[i])/splits[i] for i in range(dim)]

    
        for point in data:
            loc = []
            for d in range(dim):
                for s in range(splits[d]):
                    if (point[d] >= minima[d]+s*intervals[d] and point[d] <= minima[d] + (s+1)*intervals[d]):
                        loc.append(s)
                        break
                else:
                    loc.append(splits[d]-1)
            if dim == 3 or dim == 6:
                KDTree[loc[0]][loc[1]][loc[2]].append(point)
            if dim == 2 or dim == 4:
                KDTree[loc[0]][loc[1]].append(point)
            point.append(loc)
            
    #friends stores the whole cluster structure, cluster 0 are the points that
    #do not belong to a cluster yet
    #current_friends stores solely the friends of the current point
    friends = [data[:]]
    current_friends = []
    #Function to calculate the distance squared
    if metric == "euclid":
        def metric(a, b, d2):
            d = sum([(a[i]-b[i])**2 for i in range(dim)])
            if d > d2 or d == 0:
                return False
            else: 
                return True
                   
    
    def foaf(initial, friends, d2, cluster_nr, depth):
        loc = initial[dim]
        #
        try:
            del current_friends[depth:]
        except IndexError:
            pass
        
        current_friends.append([])
        
        #Find the values in the square [d,d] around the center point, those are
        #candidates of being friends with the current point
        candidates = []
        if dim == 3 or dim == 6:
            for box in neighbouring_boxes:
                try:
                    candidates.extend(KDTree[loc[0]+box[0]][loc[1]+box[1]][loc[2]+box[2]])
                except IndexError:
                    pass
                        
        if dim == 2 or dim == 4:
            for box in neighbouring_boxes:
                try:
                    candidates.extend(KDTree[loc[0]+box[0]][loc[1]+box[1]])
                except IndexError:
                    pass
                        
        #for new point make a new cluster, may be empty if there are no friends
        try:
            friends[cluster_nr]
        except IndexError:
            friends.append([])
        
        candidates.remove(initial)
        dens = len(candidates)
        del initial[dim]
        candidates = copy.deepcopy(candidates)
        
        if dens > density:
            f = list(map(metric, candidates, [initial for i in range(dens)], d2))            
            fn = [candidates[i] for i in range(dens) if f[i] == True]
            for c in fn:
                if c in friends[0]:
                    friends[0].remove(c)  #remove from unclustered set
                    current_friends[depth].append(c)
                    friends[cluster_nr].append(c) #add to current cluster
                
          
            #use the foaf function recursively on all current friends

            try:
                for friend in current_friends[depth]:
                    foaf(friend, friends, d2, cluster_nr, depth+1)
            except IndexError:
                pass
        
    depth = 0 #counts how deep the recursive function goes
   
   
    #run through every point that is not yet clustered, friends get changed 
    #inside the foaf function, if a point gets clustered within
    for loner in friends[0]:
        print(len(friends[0]), len(data))
        foaf(loner, friends, d2, cluster_nr, depth)
        cluster_nr += 1


    #delete empty clusters, where no friends are present
    i = 0
    while i < len(friends):
        try:
            if friends[i] == []:
                del friends[i]
            else:
                i += 1
        except IndexError:
            pass
    return friends    


d = 12 #Firend of a friend distance
lengths = [2500, 5000, 10000, 25000]
with open("ResultsExt2.txt", "w") as f:
    f.write("Lengths: {} \n ".format(lengths))
    
for datalength in lengths:
    start = time.clock()
    np.random.seed(3)
    np.random.shuffle(positions)
    data = [[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]] for pos in positions[:datalength]]

    dim = len(data[0])
    dataset = [[pos[i] for pos in data] for i in range(dim)]
    minima = [min(dataset[i]) for i in range(dim) ]
    maxima = [max(dataset[i]) for i in range(dim)]
    
    if datalength > 3000:
        splits = [int(3*(maxima[i] - minima[i])/d) for i in range(dim)]
    else:
        splits = [int(2*(maxima[i] - minima[i])/d) for i in range(dim)]
    
    intervals = [(maxima[i]-minima[i])/splits[i] for i in range(dim)]
    
    end1 = time.clock() - start
    
    start = time.clock()
    clusters = FOAF(data, d, density = 2, splits=splits)
    nr_clustersSMALL = len(clusters)
    clustersClear = []
    for i in range(len(clusters)):
        if len(clusters[i]) > 20:
            clustersClear.append(clusters[i])
    nr_clustersCleared = len(clustersClear)
    cluster_points = []
    for cluster in clustersClear:
        cluster_points.append(len(cluster))
    end2 = time.clock() - start

    with open("ResultsExt2.txt", "a") as f:
        f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {}, #ClustersDirect: {}, #ClustersCleared: {}, Points per cluster: {}, Splits: {} \n\n".format(datalength, end1, end2, nr_clustersSMALL, nr_clustersCleared, cluster_points, splits))


print("Plotting...")
colors = ["#00ffff", "#ff00ff", "#99ff33", "#3333ff", "#ffff99", "#993333", "#ccffcc", "r", "g", "c", "m", "b"] * 700
colors.insert(0, "k")


#
#if len(data[0]) == 2 or len(data[0]) == 4:
#    plt.figure()
#    plt.xlim(minima[0], maxima[0])
#    plt.ylim(minima[1], maxima[1])
#    for i in range(len(clusters)):
#        for point in clusters[i]:
#            plt.plot(point[0], point[1], marker = "o", c = "{}".format(colors[i]))
#            
#    for i in range(splits[0]):
#        plt.plot([minima[0]+(i+1)*intervals[0], minima[0]+(i+1)*intervals[0]], [minima[1], maxima[1]], "k")
#        
#    for i in range(splits[1]):
#        plt.plot([minima[0], maxima[0]], [minima[1]+(i+1)*intervals[1], minima[1]+(i+1)*intervals[1]], "k")
#    
#
#if len(data[0]) == 3 or len(data[0]) == 6:
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    for i in range(len(clustersClear)):
#        for point in clustersClear[i]:
#            ax.scatter(point[0], point[1], point[2], c = "{}".format(colors[i]))
            