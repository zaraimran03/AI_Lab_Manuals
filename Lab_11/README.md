# Lab 11: Agglomerative Hierarchical Clustering

## Lab Overview
Implementation of Agglomerative Hierarchical Clustering with dendrograms and linkage methods.

## Objectives
- Understand hierarchical clustering
- Apply different linkage methods
- Interpret dendrograms
- Compare with K-Means

## Tasks Completed

### Q1: Different Linkage Methods
- Loaded shopping-data.csv
- Extracted Annual Income and Spending Score
- Applied linkage methods: ward, complete, average
- Plotted clusters for each method
- Compared cluster structures

### Q2: Dendrogram Analysis
- Drew dendrogram using ward linkage
- Identified optimal number of clusters visually
- Determined merge heights
- Interpreted cluster hierarchy

### Q3: Agglomerative vs Divisive
- Created synthetic dataset (10-12 points)
- Plotted Agglomerative dendrogram
- Simulated Divisive clustering (manual/KMeans)
- Compared merge/split patterns
- Discussed computational differences

## Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
import seaborn as sns
```

## Key Learnings
- Bottom-up vs top-down clustering
- Linkage methods (ward, complete, average, single)
- Dendrogram interpretation
- Optimal cluster selection
- Agglomerative more common in practice
- Applications in market segmentation