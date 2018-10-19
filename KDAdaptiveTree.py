import numpy as np
import matplotlib.pyplot as plt
import time
import csv

start = time.clock()
with open("/home/thomas/Uni/3.Jahr/FS17/DataScience/Data/cdm2.00100.578872.halo.ascii", "r") as f:
    #maximum = 2780000
    nr_particles = 278000
    nr_lines = int(nr_particles)
    i = -2
    positions = [i for i in range(nr_lines)]
    while (i < nr_lines):
        if i > -1:
            positions[i] = [float(entry) for entry in f.readline().split()]
        else:
            f.readline()
        if i%10000 == 0:
            print(i)
        i += 1

read_in_time = time.clock() - start

#with open("Datafile6D.csv", "r") as f:
#    nr_lines = 1000
#    i = 0
#    positions = []  
#    csv_reader = csv.reader(f)
#    for line in csv_reader:
#        if i < nr_lines:
#            positions.append([float(line[j]) for j in range(len(line))])   
#            
#        i += 1

def AKDSpatialTree(data, partsperbox):
    tree = []
    counter = 0
    dim = len(data[0])
    spatial_dim = int(dim / 2)
    global counter
    def AKD(data, tree, partsperbox, __dim = 0):
        l = len(data)
        global counter
        if l <= partsperbox:
            tree.append(np.array(data))
            counter += 1
        else:
            median = int(l/2)
            data = sorted(data, key = lambda x: x[__dim%spatial_dim])
            AKD(data[:median],  tree, partsperbox, __dim+1)
            AKD(data[median:], tree, partsperbox, __dim+1)
        
    AKD(data, tree, partsperbox)
    
    cubesides = [i for i in range(len(tree))]
    cubevolumes = [i for i in range(len(tree))]
    density = [i for i in range(len(tree))]
    
    for i in cubesides:
        cubesides[i] = [[min(tree[i][:,d]), max(tree[i][:,d])] for d in range(spatial_dim)] 
        cubevolumes[i] = np.prod([m[1]-m[0] for m in cubesides[i]])
        density[i] = len(tree[i])/cubevolumes[i]
        
     
    print(max(density))
    return tree
    
    
    
start = time.clock() 
clusters = AKDSpatialTree(positions, partsperbox=10)
summe = 0

KDTree_time = time.clock()-start

print(KDTree_time)