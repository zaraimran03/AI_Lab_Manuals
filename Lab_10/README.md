# Lab 10: K-Means Clustering Schemes in Machine Learning

## Lab Overview
Implementation of K-Means clustering algorithm from scratch and using scikit-learn.

## Objectives
- Understand unsupervised learning
- Implement K-Means manually
- Use scikit-learn KMeans
- Determine optimal K using Elbow Method

## Tasks Completed

### Q1: K-Means from Scratch
- Implemented K-Means on 9 points
- Used K=3 clusters
- Initialized centroids: C1=P7, C2=P9, C3=P8
- Performed 2 manual iterations
- Plotted points and centroids
- Labeled all points (P1…P9)

### Q2: Scikit-learn KMeans
- Used same 9 points
- Tested K = 2, 3, 4
- Plotted clustering for each K
- Compared cluster sizes and centroids
- Analyzed shape changes with K

### Q3: Add New Point and Re-Cluster
- Added P10(6,2) to dataset
- Ran K-Means with K=3
- Identified which cluster P10 joins
- Analyzed centroid shifts
- Sketched before/after clusters

### Q4: Distance Table + First Iteration
- Computed Euclidean distances manually
- Created distance table
- Performed first iteration only
- Calculated new centroids
- Plotted first-iteration graph

## Libraries Used
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

## Key Learnings
- K-Means algorithm steps
- Euclidean distance calculation
- Centroid update process
- Elbow method for optimal K
- Effect of new data points on clustering
- Comparison: manual vs library implementation