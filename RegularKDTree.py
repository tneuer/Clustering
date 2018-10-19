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
    #maximum = 1000000
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
 

def KDSpatialTree(data, splits, min_cell_dens = -1, min_cluster_size = 0):
    """
    Most easiest spatial density finder, by making splits[0]*splits[1]*splits[2]
    different regular boxes and looking for density maxima. 
    
    Input:  data    -   List of tuples or lists containing the coordinates
                        (x,y,z,vx,vy,vz,..) of a point 
            splits  -   Typically 3D list, with number of slices on x,y,z axis
            min_cluster_density - Density constraint on cluster for break                                   
            
    Output: clusters -  List of clusters, that contain all the points.
                        The last cluster are the unclustered points
            stats    -  Important statistical values of the overall density
    """
    
    dim = len(data[0])
    #array of length dim, every list contains one coordinate
    dataset = [np.array([pos[i] for pos in data]) for i in range(dim)]
    
    print("Building tree...")
    #Creation of kD-Tree
    if splits != 0:
        #Creating empty tree
        if dim == 3 or dim == 6:
            spatial_dim = 3
            KDTree = [[[[] for i in range(splits[2])] for j in range(splits[1])] for l in range(splits[0])]
            neighbouring_boxes = [[-1,-1,-1], [-1,-1,0], [-1,-1,1], [-1,0,-1], [-1,0,0], [-1,0,1], [-1,1,-1], [-1,1,0], [-1,1,1], [0,-1,-1], [0,-1,0], [0,-1,1], [0,0,-1], [0,0,1], [0,1,-1], [0,1,0], [0,1,1], [1,-1,-1], [1,-1,0], [1,-1,1], [1,0,-1], [1,0,0], [1,0,1], [1,1,-1], [1,1,0], [1,1,1]]
        if dim == 2 or dim == 4:
            neighbouring_boxes = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]]
            spatial_dim = 2
            KDTree = [[[] for j in range(splits[1])] for l in range(splits[0])]
            
        #Determining width of a cell for every dimension
        minima = [min(dataset[i]) for i in range(spatial_dim) ]
        maxima = [max(dataset[i]) for i in range(spatial_dim)]
        intervals = [(maxima[i]-minima[i])/splits[i] for i in range(spatial_dim)]

        #Assgning each point to a cell
        for point in data:
            loc = []
            for d in range(spatial_dim):
                for s in range(splits[d]):
                    if (point[d] >= minima[d]+s*intervals[d] and point[d] <= minima[d] + (s+1)*intervals[d]):
                        loc.append(s)
                        break
                else:
                    loc.append(splits[d]-1)
              
            #Append point index to cell
            if dim == 3 or dim == 6:
                KDTree[loc[0]][loc[1]][loc[2]].append(point)
            if dim == 2 or dim == 4:
                KDTree[loc[0]][loc[1]].append(point)
         
    
    #Inserting number of points in a cell and its location in a list
    dens = []
    for i in range(splits[0]):
        for j in range(splits[1]):
            for k in range(splits[2]):
                dens.append([len(KDTree[i][j][k]), i, j, k])
    
    dens = sorted(dens)
    dens = np.array(dens)
    
    #Filtering out background via minimum density condition
    densloc = dens[np.where(dens[:,0] >= min_cell_dens)]
    bgloc = dens[np.where(dens[:,0] < min_cell_dens)][:,1:]
    bg = []
    for loc in bgloc:
        bg.extend(KDTree[loc[0]][loc[1]][loc[2]])
    
    #Determining useful statistics about the densities alone (w/o position)
    dens = dens[:,0]
    dens_stats = [("min", min(dens)), ("5%", np.round(np.percentile(dens, 5), 2)), ("50%", np.median(dens)), ("95%", np.round(np.percentile(dens, 95), 2)), ("max", max(dens)), ("mean", np.round(np.mean(dens), 2)), ("std", np.round(np.std(dens), 2))]
    densloc = [list(loc) for loc in densloc[:,1:]]
    
    clusters = []
    nr_cells_per_cluster = []
    cl_nr = 0
    print("Find Cluster...")
    
    #Start of density peak finder
    while (len(densloc) > 0):
        
        #Take current density peak
        mx, my, mz = densloc.pop()
        max_cell = KDTree[mx][my][mz]
        
        #Create new cluster
        clusters.append(max_cell)
        nr_cells_per_cluster.append(1)
        
        #Find all surrounding cells if not already assigned (i.d. still in densloc)
        candidate_cells = np.array([mx,my,mz]) + neighbouring_boxes            
        candidate_cells = [list(loc) for loc in candidate_cells if list(loc) in densloc]     

        #Look in the surrounding boxes
        while (len(candidate_cells) > 0):
            
            #Do the same as before but with the Friend-Cells
            next_loc = candidate_cells.pop()
            next_cell = KDTree[next_loc[0]][next_loc[1]][next_loc[2]]
            
            #Add next cell to the same cluster
            clusters[cl_nr].extend(next_cell)
            nr_cells_per_cluster[cl_nr] += 1
            densloc.remove(next_loc)
            
            #Find friend cells of friend cell
            surrounding = np.array(next_loc) + neighbouring_boxes
            surrounding = [list(loc) for loc in surrounding]
            surrounding = [loc for loc in surrounding if (loc in densloc and loc not in candidate_cells)]
            candidate_cells.extend(surrounding)
            
        cl_nr += 1  
    
    #Delete Clusters, that are too small. Append to background
    deleter = []
    del_counter = 0
    for i in range(len(clusters)):
        if len(clusters[i]) < min_cluster_size:
            bg.extend(clusters[i])
            deleter.append(i-del_counter)
            del_counter += 1
            
    for d in deleter:
        del clusters[d]
        del nr_cells_per_cluster[d]
    
    #append background
    clusters.append(bg)   
    nr_cells_per_cluster.append(len(bgloc))    
    dens_stats.append(("Cells per cluster", nr_cells_per_cluster))
    return clusters, dens_stats

            
#Set procedure  
lengths = [10000]
doc_name = "Useless.txt"
cell_dens = 30
clu_size = 50

with open(doc_name, "w") as f:
    f.write("Lengths: {} \n ".format(lengths))
    
for datalength in lengths:
    print(datalength)
    start = time.clock()
    np.random.seed(3)
    np.random.shuffle(positions)
    data = [[pos[0], pos[1], pos[2]] for pos in positions[:datalength]]

    dim = len(data[0])
    
    end1 = time.clock() - start
    
    dataset = [[pos[i] for pos in data] for i in range(dim)]
    minima = [min(dataset[i]) for i in range(dim) ]
    maxima = [max(dataset[i]) for i in range(dim)]
    
    splits = [50, 50, 50]
    
    start = time.clock()
    
    clusters, stats = KDSpatialTree(data, splits = splits, min_cell_dens = cell_dens, min_cluster_size = clu_size)        
    print(len(clusters))              
    
    avg_dens_tot = datalength/np.prod(splits)
    avg_dens_cluster = [ len(clusters[i])/stats[-1][1][i] for i in range(len(clusters))]

    cluster_points = []
    summe = 0
    for cluster in clusters:
        cluster_points.append(len(cluster))
        summe += len(cluster)
    
    print(summe)
    end2 = time.clock() - start

    with open(doc_name, "a") as f:
        f.write("Points: {}, TimeSetUp: {}, TimeAlgorithm: {} \n\n Param: [splits: {}, min_cell_d: {} min_cl_size: {}] \n\n #Clusters: {}, Points per cluster: {} \n\n Avg points per cell Total: {} \n Average points per cell Cluster: {} \n\n Density Stats: {}\n\n\n\n".format(datalength, end1, end2, splits, cell_dens, clu_size, len(clusters), cluster_points, avg_dens_tot, avg_dens_cluster, stats))
        

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
        plt.plot(x, y, z, "ok", alpha = 0.005, ms = 2)
    col += 1
#plt.show()
print(time.clock() - start)
        

############################################################################
"""""""""
    2D Plot Grid
"""""""""
############################################################################


start = time.clock()
fig = plt.figure()
ax = plt.gca()
ax.set_axis_bgcolor((0, 0, 0))
col = 1
intervals = [(maxima[i]-minima[i])/splits[i] for i in range(dim)]

for cluster in clusters:
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    if col != len(clusters):
        plt.scatter(x, y, s = 7, c = "{}".format(colors[col]))
    else:
        plt.scatter(x, y, s = 2, c = "w", alpha = 0.2)
    col += 1
    
for i in range(-1,splits[0]):
    plt.plot([minima[0]+(i+1)*intervals[0], minima[0]+(i+1)*intervals[0]], [minima[1], maxima[1]], "y", alpha = 0.2)
        
for i in range(-1,splits[1]):
    plt.plot([minima[0], maxima[0]], [minima[1]+(i+1)*intervals[1], minima[1]+(i+1)*intervals[1]], "y", alpha = 0.2)
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
intervals = [(maxima[i]-minima[i])/splits[i] for i in range(dim)]

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
        plt.scatter(x, y, s = 2, c = "k", alpha = 0.2)
    it += 1
    
for i in range(-1,splits[0]):
    plt.plot([minima[0]+(i+1)*intervals[0], minima[0]+(i+1)*intervals[0]], [minima[1], maxima[1]], "y", alpha = 0.3)
        
for i in range(-1,splits[1]):
    plt.plot([minima[0], maxima[0]], [minima[1]+(i+1)*intervals[1], minima[1]+(i+1)*intervals[1]], "y", alpha = 0.3)
            
    
print(time.clock() - start)
plt.show()