# Lab 3: Decision Tree Classifier

**Student Name:** Zara Imran  
**SAP ID:** 70149236  
**Section:** BSIET-5I

## Lab Overview
Implementation of Decision Tree Classifier for classification tasks with manual entropy/information gain calculations and sklearn implementation.

## Objectives
- Understand decision tree construction
- Calculate entropy and information gain manually
- Implement decision trees using sklearn
- Visualize decision trees

## Tasks Completed

### Q1: Entropy and Information Gain (Manual Calculation)
- Calculated entropy of target variable (Result)
- Computed information gain for attribute StudyHours
- Determined root node based on maximum information gain

### Q2: Decision Tree on Small Dataset
- Created DataFrame with student data
- Used LabelEncoder for categorical variables
- Trained DecisionTreeClassifier with entropy criterion
- Visualized tree using plot_tree()

### Q3: Decision Tree on Iris Dataset
- Loaded Iris dataset
- Split into 70% training, 30% testing
- Trained decision tree with entropy criterion
- Evaluated accuracy on test set
- Identified most important feature at root

### Q4: Decision Tree on MNIST Digits Dataset
- Loaded MNIST digit dataset (sklearn.datasets)
- Preprocessed 8x8 image data
- Trained Decision Tree Classifier
- Evaluated accuracy and confusion matrix
- Analyzed limitations for image data

## Libraries Used
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

## Key Learnings
- Entropy and information gain calculations
- Decision tree construction process
- Feature importance analysis
- Visualization of decision boundaries
- Limitations of decision trees for complex data