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
with open("CMSample100k.txt", "r") as f:
    #maximum = 100000
    nr_particles = 100000
    nr_lines = int(nr_particles)
    i = 0
    positions = [i for i in range(nr_lines)]
    while (i < nr_lines):
        positions[i] = [float(entry) for entry in f.readline().split()]
        if i%10000 == 0:
            print(i)
        i += 1

read_in_time = time.clock() - start  
 

def FOAFITERTREE(data, d, min_point_density = 0, min_cluster_dens = 0, stats = False, splits = 0):
    """
    Iteratively uses the friends of friends algorithm to look for clusters.
    
    Input:  data    -   List of tuples or lists containing the coordinates
                        (x,y,z,vx,vy,vz,..) of a point 
            dist    -   friends radius for the metric
            min_point_density   - Density constraint on points for acceptance
            min_cluster_density - Density constraint on cluster for acceptance
            stats - Gives out more stats but increases computational time 
                    dramaticall. Should only be used with < 10000 particles                                    
            
    Output: friends -   List of clusters, that contain all the points.
                        The last cluster are the unclustered points
            if stats = T: data will be altered, 3 values get appended
            last: avg distance within cluster
            second last: average distance total
            third last: number of particles within FOF radius
    """
    
    dim = len(data[0])
    #array of length dim, every list contains one coordinate
    dataset = [np.array([pos[i] for pos in data]) for i in range(dim)]
    nr_points = len(dataset[0])
    d2 = d**2 #reduces computational time, no sqrt necessary
    
    #ID for each particle
    particles = [i for i in range(nr_points)]
    particlesLoc = [i for i in range(nr_points)]
    
    #One group for every particle in the beginning
    groups = np.zeros((nr_points,1)).tolist()
    
    #Creation of kD-Tree
    if splits != 0:
        #Creating empty tree
        if dim == 3 or dim == 6:
            spatial_dim = 3
            KDTree = [[[[] for i in range(splits[2])] for j in range(splits[1])] for l in range(splits[0])]
            neighbouring_boxes = [[-1,-1,-1], [-1,-1,0], [-1,-1,1], [-1,0,-1], [-1,0,0], [-1,0,1], [-1,1,-1], [-1,1,0], [-1,1,1], [0,-1,-1], [0,-1,0], [0,-1,1], [0,0,-1], [0,0,0], [0,0,1], [0,1,-1], [0,1,0], [0,1,1], [1,-1,-1], [1,-1,0], [1,-1,1], [1,0,-1], [1,0,0], [1,0,1], [1,1,-1], [1,1,0], [1,1,1]]
        if dim == 2 or dim == 4:
            neighbouring_boxes = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
            spatial_dim = 2
            KDTree = [[[] for j in range(splits[1])] for l in range(splits[0])]
            
        #Determining width of a cell for every dimension
        minima = [min(dataset[i]) for i in range(spatial_dim) ]
        maxima = [max(dataset[i]) for i in range(spatial_dim)]
        intervals = [(maxima[i]-minima[i])/splits[i] for i in range(spatial_dim)]

        #Assgning each point to a cell
        for i in range(nr_points):
            loc = []
            for d in range(spatial_dim):
                for s in range(splits[d]):
                    if (data[i][d] >= minima[d]+s*intervals[d] and data[i][d] <= minima[d] + (s+1)*intervals[d]):
                        loc.append(s)
                        break
                else:
                    loc.append(splits[d]-1)
              
            #Append point index to cell
            if dim == 3 or dim == 6:
                KDTree[loc[0]][loc[1]][loc[2]].append(i)
            if dim == 2 or dim == 4:
                KDTree[loc[0]][loc[1]].append(i)
            particlesLoc[i] = loc
    
    while len(particles)>0:
          # remove the particle from the particles list
          index = particles.pop()
          loc = particlesLoc[index]
          groups[index] = [index]
          print("#N ", index)
          
          #Find particles in surrounding cells
          cells = []
          if dim == 3 or dim == 6:
              for box in neighbouring_boxes:
                   try:
                       cells.extend(KDTree[loc[0]+box[0]][loc[1]+box[1]][loc[2]+box[2]])
                   except IndexError:
                       pass
          
          #Calculate the absolute value of the distance vectors
          diff = [(dataset[i][cells] - dataset[i][index])**2 for i in range(dim)]
          dr = diff[0]
          for i in range(dim-1):
              dr += diff[i+1]
        
          #Find friends of the current point
          id_to_look = list(np.array(cells)[np.where(dr < d2)])
          id_to_look.remove(index)
          
          #Calculate the "density" and the total avg distance
          if stats:
              data[index].append(len(id_to_look))
              data[index].append(sum(np.sqrt(dr))/(nr_points-1))
              
          # remove all the neighbors from the particles list
          for i in id_to_look:
                if (i in particles):
                   particles.remove(i)
                   
          #Add friends to current index
          groups[index] = groups[index] + id_to_look
          new_nlist = id_to_look
          
          #Fiend friends of friends in this while loop, same structure as before
          if len(id_to_look) > min_point_density:
              while len(new_nlist)>0:
                      index_n = new_nlist.pop()
                      loc = particlesLoc[index_n]
                      
                      #Find particles in surrounding cells
                      cells = []
                      if dim == 3 or dim == 6:
                          for box in neighbouring_boxes:
                               try:
                                   cells.extend(KDTree[loc[0]+box[0]][loc[1]+box[1]][loc[2]+box[2]])
                               except IndexError:
                                   pass
                      
                      diff = [(dataset[i][cells] - dataset[i][index_n])**2 for i in range(dim)]
                      dr = diff[0]
                      for i in range(dim-1):
                          dr += diff[i+1]
                          
                      id_to_look = list(np.array(cells)[np.where(dr < d2)])
                      
                      if stats:
                          data[index_n].append(len(id_to_look))
                          data[index_n].append(sum(np.sqrt(dr))/(nr_points-1))
                      
                      if  len(id_to_look) > min_point_density:
                          #add friends of friend to the list 
                          id_to_look = list(set(id_to_look) & set(particles))
                          groups[index] = groups[index] + id_to_look
                          new_nlist = new_nlist + id_to_look
                          for k in id_to_look:
                              particles.remove(k)
                              
          
    #Filter empty clusters
    groups = list(filter(([0.0]). __ne__, groups))
    groups.append([])
    deleter = []
    del_counter = 0
    l = len(groups)
    
    #Store the corrected indices for too small clusters in deleter
    for i in range(l-1):
        if len(groups[i]) < min_cluster_dens:
            groups[l-1].append(groups[i])
            deleter.append(i-del_counter)
            del_counter += 1
    
    #Delete those elements selected above
    for d in deleter:
        del groups[d]
        
    #Flatten the last unclustered list, currently list of lists
    l = len(groups)
    groups[l-1] = [ID for subgroup in groups[l-1] for ID in subgroup]  
     
    #Delete selected groups
    if groups[l-1] == []:
        del groups[l-1]
     
    #Calculate avg distance within a cluster
    if stats:
        avg_dists = [ [i, sum([np.sqrt(sum((np.array(data[i][:dim]) - np.array(data[j][:dim]))**2)) for j in group])/(len(group)-1) ] for group in groups for i in group]
        for (point, dist) in avg_dists:
            data[point].append(dist)
    
    
    #Map indices to their points
    clusters = [[data[i] for i in group] for group in groups]
    return clusters


#Set procedure  
lengths = [5000, 10000, 25000, 50000, 75000, 100000]
S = False
d = 0.004
doc_name = "CM100kFOAFSpatialTree.txt"

with open(doc_name, "w") as f:
    f.write("Lengths: {} \n ".format(lengths))
    
for datalength in lengths:
    start = time.clock()
    np.random.seed(3)
    np.random.shuffle(positions)
    data = [[pos[0], pos[1], pos[2]] for pos in positions[:datalength]]

    dim = len(data[0])
    
    end1 = time.clock() - start
    
    dataset = [[pos[i] for pos in data] for i in range(dim)]
    minima = [min(dataset[i]) for i in range(dim) ]
    maxima = [max(dataset[i]) for i in range(dim)]
    
    splits = [2*int((maxima[i] - minima[i])/d) for i in range(dim)]
    
    start = time.clock()
    
    clusters = FOAFITERTREE(data, d = d, min_point_density = datalength/300, min_cluster_dens = 200, stats = S, splits = splits)
    print(len(clusters))
    if S:
        avg_dens_tot = sum([point[len(point)-3] for point in data])/len(data)
        avg_dens_cluster = [ sum([point[len(point)-3] for point in cluster])/len(cluster) for cluster in clusters]
        avg_dist_tot = sum([point[len(point)-2] for point in data])/len(data)
        avg_dist_cluster = [sum([point[len(point)-1] for point in cluster])/len(cluster) for cluster in clusters]

    cluster_points = []
    for cluster in clusters:
        cluster_points.append(len(cluster))
    end2 = time.clock() - start

    if S:
        with open(doc_name, "a") as f:
            f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {}, #Clusters: {}, Points per cluster: {}, Average Density Total: {}, Average Density Cluster: {}, Average Distance Total: {}, Average Distance Cluster: {}\n\n".format(datalength, end1, end2, len(clusters), cluster_points, avg_dens_tot, avg_dens_cluster, avg_dist_tot, avg_dist_cluster))

    else:
        with open(doc_name, "a") as f:
            f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {}, #Clusters: {}, Points per cluster: {}\n\n".format(datalength, end1, end2, len(clusters), cluster_points))

        

print("Plotting...")
colors = ["#00ffff", "#ff00ff", "#99ff33", "#3333ff", "#ffff99", "#993333", "#ccffcc", "r", "g", "c", "m", "b"] * 700

start = time.clock()
fig = plt.figure()
ax = Axes3D(fig) 
ax.set_axis_bgcolor((0, 0, 0))
col = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    z = [point[2] for point in cluster]
    if col != len(clusters):
        plt.plot(x, y, z, marker = "o", c = "{}".format(colors[col]), ms = 3)
    else:
        plt.plot(x, y, z, marker = "o", c = "w", alpha = 0.3, ms = 2)
    col += 1
    
print(time.clock() - start)
        

start = time.clock()
fig = plt.figure()
ax = plt.gca()
ax.set_axis_bgcolor((0, 0, 0))
col = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    if col != len(clusters):
        plt.scatter(x, y, s = 10, c = "{}".format(colors[col]))
    else:
        plt.scatter(x, y, s = 5, c = "w", alpha = 0.15)
    col += 1
    
print(time.clock() - start)  
            