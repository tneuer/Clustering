# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:54:42 2017

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv

start = time.clock()
with open("/home/thomas/Uni/3.Jahr/FS17/DataScience/Data/cdm2.00100.578872.halo.ascii", "r") as f:
    #maximum = 2780000
    nr_particles = 2780000
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

nr_particles2 = 100000
seed = 13
np.random.seed(seed)
np.random.shuffle(positions)
positions = positions[:nr_particles2]
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
z = [pos[2] for pos in positions]
vx = [pos[3] for pos in positions]
vy = [pos[4] for pos in positions]
vz = [pos[5] for pos in positions]

plt.figure()
plt.scatter(x, y, s = 0.001)
plt.title("#Particles: {} / {}, Seed: {}".format(nr_particles2, nr_particles, seed))


plt.show()

#with open("Sample1M.txt", "w") as f:
#    for i in range(nr_particles2):
#        f.write("{} {} {} {} {} {}\n".format(positions[i][0], positions[i][1], positions[i][2], positions[i][3], positions[i][4], positions[i][5]))



"""
    Creation of spirale data
"""


        