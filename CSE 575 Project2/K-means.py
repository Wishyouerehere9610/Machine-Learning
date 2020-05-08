# CSE575 Project2 Unsupervised Learning (K-means)
# Ziming Dong

import scipy.io as scio
import numpy as np
import random
import math as m
import sys
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# To make plot clear, set the figuresieze and plot style.
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

dataFile = 'AllSamples.mat'
# read data.mat file
data = scio.loadmat(dataFile)

dataset = data['AllSamples']

# initial x,y coordinates from extracting value from All Samples` arrary
# x=[]
# y=[]

# for row in dataset:
#    x.append(row[0])

# for row in dataset:
#   y.append(row[1])

# Plot 300 points from dataset, see what it looks like
# for i in range(len(x)):
#    f1=x[i]
#    f2=y[i]
#    plt.scatter(f1, f2, c='black')

sorted(data.keys())
n = dataset.shape[1]


# Implement calculate distance function between two points.
def dist(a, b):
    return np.linalg.norm(a - b)


# Strategy 1: Random pick the initial centers from the given samples.
def s1(dataset, k):
    i = 0
    center = np.zeros((k, 2))
    while (i < k):
        center[i][0] = dataset[random.randint(0, dataset.shape[0]) - 1][0]
        center[i][1] = dataset[random.randint(0, dataset.shape[0]) - 1][1]
        i += 1
    return center


# Strategy 2: Pick the first center randomly; for the i-th center (i>1), choose a sample
# among all possible samples) such that the average distance of this choosen one to all
# previous centers is maximal.

def s2(dataset, k):
    n = dataset.shape[1]
    center = np.zeros((k, n))
    center[0][0] = dataset[random.randint(0, 299)][0]
    center[0][1] = dataset[random.randint(0, 299)][1]
    p = 1
    while (p < k):
        maxdistance = -100
        for i in range(dataset.shape[0]):
            total = 0
            for j in range(0, p):
                total = total + dist(dataset[i], center[j])
            average = total / p  # Calculate average distance

            if average > maxdistance:
                maxdistance = average
                index = i
        center[p, :] = dataset[index, :]
        p += 1
    return center


# Implement the function of k-means algorithm
def km(dataset, center, k):
    cluster = np.zeros((dataset.shape[0], 2))
    change = True
    while change:
        change = False
        for i in range(dataset.shape[0]):
            minDistance = sys.maxsize
            minIndex = 0
            for j in range(k):
                distance = dist(center[j], dataset[i])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j
            if cluster[i, 0] != minIndex:
                change = True  # when change = True, we will go over one more iteration.
                cluster[i, :] = minIndex, minDistance
        for n in range(k):
            points = dataset[np.nonzero(cluster[:, 0] == n)[0]]
            center[n, :] = np.mean(points, axis=0)  # return mean1,mean2,mean3,meank......
    return center, cluster


# Implement objection(cost) function by the formular
def Objection(center, cluster, k):
    sum = 0
    for i in range(k):
        count = 0
        obj = 0
        for j in cluster:
            if j[0] == i:
                obj = obj + dist(center[i], dataset[count]) ** 2
            count += 1
        sum = sum + obj
    return sum


# Plot the obejction value with k value from 2 to 10 by strategy 1.
res = [0] * 9
for k in range(2, 11):
    center = s1(dataset, k)
    center, cluster = km(dataset, center, k)
    res[k - 2] = Objection(center, cluster, k)
    # print objection value from k=2 to k=10 by strategy 1
    print('objection value with k = ', k, 'by Strategy1:')
    print(Objection(center, cluster, k))
x1 = range(2, 11)
y1 = res

plt.plot(x1, y1, 'ob')
print('')
print('                            ''Plot objective value by k-th to see tendency with Strategy 1 ')
plt.show()

# Plot the obejction value with k value from 2 to 10 by strategy 2.
res = [0] * 9
for k in range(2, 11):
    center = s2(dataset, k)
    center, cluster = km(dataset, center, k)
    res[k - 2] = Objection(center, cluster, k)
    # Plot the obejction value with k value from 2 to 10 by strategy 2.
    print('Objection value with k = ', k, 'by Strategy2:')
    print(Objection(center, cluster, k))
x1 = range(2, 11)
y1 = res

plt.plot(x1, y1, 'ob')
print('')
print('                              ''Plot objective value by k-th to see tendency with Strategy 2 ')
plt.show()