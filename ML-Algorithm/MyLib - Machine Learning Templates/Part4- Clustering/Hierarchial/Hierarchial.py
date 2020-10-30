#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:51:09 2020

@author: ziyad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DATA PREPROCESSING  

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values # For visualisation only two features are selected.

# Dendogram to compute the optimal number of clusters for the dataset
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distance between points (Euclidean)')
plt.show()

# MODEL TRAINING
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

# VISUALISATION
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()
plt.show()