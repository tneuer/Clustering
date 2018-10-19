import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#read in the data

#start = time.clock()
#with open("Datafile6D.csv", "r") as f:
#    positions = []
#    csv_reader = csv.reader(f)
#    for line in csv_reader:
#        positions.append([float(line[j]) for j in range(len(line))]) 
#read_in_time = time.clock() - start
        
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
 

def FOAFITER(data, d, min_point_density = 0, min_cluster_density = 0, stats = False, __depth = 0):
    """
    Iteratively uses the friends of friends algorithm to look for clusters.
    
    Input:  data    -   List of tuples or lists containing the coordinates
                        (x,y,z,vx,vy,vz,..) of a point 
            dist    -   friends radius for the metric
            min_point_density   - Density constraint on points for acceptance
            min_cluster_density - Density constraint on cluster for acceptance
            stats - Gives out more stats but increases computational time 
                    dramaticall. Should only be used with < 10000 particles                                    
            
    Output: spatial_clusters -  List of clusters, that contain all the points.
                                The last cluster are the unclustered points
            phase_clusters   -  If velocity data is available this will be a list
                                of clusters containing a list of subclusters
            if stats = T: data will be altered, 3 values get appended
            last: avg distance within cluster
            second last: average distance total
            third last: number of particles within FOF radius
    """
    
    #Determining the current FOAF radius (may differ for velocity)
    if isinstance(d, int) or isinstance(d, float):
        d_now = d
    elif len(d) != 1:
        d_now = d[__depth]
        
    #Determining the current min_point_density (may differ for velocity)
    if isinstance(min_point_density, int) or isinstance(min_point_density, float):
        min_point_dens_now = min_point_density
    elif len(d) != 1:
        min_point_dens_now = min_point_density[__depth]
        
    #Determining the current min_cluster_density (may differ for velocity)
    if isinstance(min_cluster_density, int) or isinstance(min_cluster_density, float):
        min_cluster_dens_now = min_cluster_density
    elif len(d) != 1:
        min_cluster_dens_now = min_cluster_density[__depth]
    
    #Determine spatial dimension    
    dim = len(data[0])
    if (dim == 4 or dim == 6) and __depth == 0:
        phasespace = True                           #Explore phase space later
        spatial_dim = int(dim/2)
    elif (dim == 4 or dim == 6) and __depth == 1:
        phasespace = False                          #Already in phase space
        spatial_dim = int(dim/2)
    else:
        phasespace = False
        spatial_dim = dim
    
    #array of length dim, every list contains one coordinate [[allx], [ally],...]
    if __depth == 0:
        dataset = [np.array([pos[i] for pos in data]) for i in range(spatial_dim)]
    else:
        dataset = [np.array([pos[i] for pos in data]) for i in range(spatial_dim,dim)]
    
    nr_points = len(dataset[0])
    d2 = d_now**2 #reduces computational time, no sqrt necessary
    
    #ID for each particle
    particles = [i for i in range(nr_points)]
    
    #One group for every particle in the beginning
    groups = np.zeros((nr_points,1)).tolist()
    
    if __depth == 0:
        print("Finding clusters in configuration space...")
    
    while len(particles)>0:
        
          start = time.clock()
          # remove the particle from the particles list
          index = particles.pop()
          groups[index] = [index]
                    
          #Calculate the absolute value of the distance vectors
          diff = [(dataset[i] - dataset[i][index])**2 for i in range(spatial_dim)]
          dr = diff[0]
          for i in range(spatial_dim-1):
              dr += diff[i+1]
          
          #Find friends of the current point
          id_to_look = np.where(dr < d2)[0].tolist()
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
          if len(id_to_look) > min_point_dens_now:
              while len(new_nlist)>0:
                      index_n = new_nlist.pop()
                      
                      diff = [(dataset[i] - dataset[i][index_n])**2 for i in range(spatial_dim)]
                      dr = diff[0]
                      for i in range(spatial_dim-1):
                          dr += diff[i+1]
                          
                      id_to_look = np.where(dr < d2)[0].tolist()
                      
                      if stats:
                          data[index_n].append(len(id_to_look))
                          data[index_n].append(sum(np.sqrt(dr))/(nr_points-1))
                      
                      if  len(id_to_look) > min_point_dens_now:
                          #add friends of friend to the list 
                          id_to_look = list(set(id_to_look) & set(particles))
                          groups[index] = groups[index] + id_to_look
                          new_nlist = new_nlist + id_to_look
                          for k in id_to_look:
                              particles.remove(k)
                              
          if (time.clock()-start) > 1:
              print(len(particles))
                              
          
    #Filter empty clusters
    groups = list(filter(([0.0]). __ne__, groups))
    groups.append([])
    deleter = []
    del_counter = 0
    l = len(groups)
    
    #Store the corrected indices for too small clusters in deleter
    for i in range(l-1):
        if len(groups[i]) < min_cluster_dens_now:
            groups[l-1].append(groups[i])
            deleter.append(i-del_counter)
            del_counter += 1
    
    #Delete those elements selected above
    for dl in deleter:
        del groups[dl]
        
    #Flatten the last unclustered list, currently list of lists
    l = len(groups)
    groups[l-1] = [ID for subgroup in groups[l-1] for ID in subgroup]  
     
    #Delete selected groups
    if groups[l-1] == []:
        del groups[l-1]
     
    #Calculate avg distance within a cluster
    if stats:
        print("Calculating stats")
        avg_dists = [ [i, sum([np.sqrt(sum((np.array(data[i][:dim]) - np.array(data[j][:dim]))**2)) for j in group])/(len(group)-1) ] for group in groups for i in group]
        for (point, dist) in avg_dists:
            data[point].append(dist)
    
    #Map indices to their points
    clusters = [[data[i] for i in group] for group in groups]
    spatial_clusters = clusters[:]
    
    #Creating subclusters
    phase_clusters = [0 for i in range(len(clusters))]
    empty = [0 for i in range(len(clusters))]
    
    #Calling the algorithm on the velocity space to find substrctures
    if phasespace:
        print("Looking for phase space structure...")
        for i in range(len(clusters)):
            print("Cluster ", i+1, " of ", len(clusters))
            phase_clusters[i] = FOAFITER(clusters[i], d, min_point_density, min_cluster_density, stats = False, __depth = 1)
    
    if phase_clusters == empty:
        return spatial_clusters
    else:
        return spatial_clusters, phase_clusters


#Set procedure  
#lengths = [5000, 10000, 25000, 50000, 75000, 100000]
lengths = [50000]
S = False
d = [0.001, 0.07]
min_p = [20, 10]
min_cl = [20, 10]

doc_name = "Test.txt"

with open(doc_name, "w") as f:
    f.write("Lengths: {} \n ".format(lengths))
    
for datalength in lengths:
    start = time.clock()
    np.random.seed(3)
    np.random.shuffle(positions)
    data = [[pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]] for pos in positions[:datalength]]

    dim = len(data[0])
    
    end1 = time.clock() - start
    
    start = time.clock()
    
    clusters, phase_clusters = FOAFITER(data, d = d, min_point_density = min_p, min_cluster_density = min_cl, stats = S)
    print(len(clusters), sum([len(s) for s in phase_clusters]) )
    
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
            f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {}, d: {}, #Clusters: {}, Points per cluster: {}, Average Density Total: {}, Average Density Cluster: {}, Average Distance Total: {}, Average Distance Cluster: {}\n\n".format(datalength, end1, end2, d, len(clusters), cluster_points, avg_dens_tot, avg_dens_cluster, avg_dist_tot, avg_dist_cluster))

    else:
        with open(doc_name, "a") as f:
            f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {} \n\n Param: [d: {}, min_p_d: {} min_cl_d: {}] \n\n #Clusters: {}, Points per cluster: {}\n\n\n\n".format(datalength, end1, end2, d, min_p, min_cl, len(clusters), cluster_points))
        

print("Plotting...")
colors = ["#00ffff", "#ff00ff", "#99ff33", "#3333ff", "#ffff99", "#993333", "#ccffcc", "r", "g", "c", "m", "b"] * 700

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
    if col != len(clusters):
        plt.plot(x, y, z, marker = "o", c = "{}".format(colors[col]), ms = 3)
    else:
        plt.plot(x, y, z, "ok", alpha = 0.05, ms = 2)
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
    if col != len(clusters):
        plt.scatter(x, y, s = 7, c = "{}".format(colors[col]))
    else:
        plt.scatter(x, y, s = 2, c = "w", alpha = 0.2)
    col += 1
    
#plt.show()
print(time.clock() - start)


############################################################################
"""""""""
    2D Plot circles
"""""""""
############################################################################

if dim == 4 or dim == 6:
    start = time.clock()
    fig = plt.figure()
    ax = plt.gca()
    it = 1
    
    for cluster in phase_clusters:
        for subcluster in cluster:
            x = [point[0] for point in subcluster]
            y = [point[1] for point in subcluster]
            cen = (np.mean(x), np.mean(y))
            xdist = max(x - cen[0])
            ydist = max(y - cen[1])
            rad = np.sqrt(xdist**2 + ydist**2)
            circle = plt.Circle(cen, rad, color = "b", fill = False)
            if it != len(phase_clusters):
                ax.add_artist(circle)
        it += 1
                    
        
    print(time.clock() - start)


else:
    fig = plt.figure()
    ax = plt.gca()


start = time.clock()
it = 1

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    cen = (np.mean(x), np.mean(y))
    xdist = max(x - cen[0])
    ydist = max(y - cen[1])
    rad = np.sqrt(xdist**2 + ydist**2)
    circle = plt.Circle(cen, rad, color = "r", fill = False)
    if it != len(clusters):
        plt.scatter(x, y, c = "k", s = 7)
        ax.add_artist(circle)
    else:
        plt.scatter(x, y, s = 2, c = "k", alpha = 0.8)
    it += 1
                
    
print(time.clock() - start)


plt.show()