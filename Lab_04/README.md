# Lab 4: Random Forest and Support Vector Machine Classifier

## Lab Overview
Implementation and comparison of Random Forest and SVM classifiers on various datasets.

## Objectives
- Understand ensemble learning (Random Forest)
- Understand SVM and kernel methods
- Compare performance of both algorithms
- Apply to real-world datasets

## Tasks Completed

### Q1: Random Forest on Iris Dataset
- Loaded Iris dataset
- Split into 70% training, 30% testing
- Trained RandomForestClassifier
- Predicted flower species
- Calculated accuracy

### Q2: SVM on Breast Cancer Dataset
- Loaded Breast Cancer dataset
- Trained SVM with linear kernel
- Evaluated using accuracy and confusion matrix

### Q3: Random Forest on Custom CSV Dataset
- Loaded students.csv (study_hours, attendance, marks, result)
- Trained Random Forest to predict Pass/Fail
- Displayed accuracy and feature importance

### Q4: SVM on Digits Dataset
- Loaded Handwritten Digits dataset
- Trained SVM with RBF kernel
- Evaluated accuracy
- Visualized misclassified samples

### Q5: Comparison – Random Forest vs SVM
- Used Wine dataset
- Trained both models
- Compared accuracies
- Concluded which performs better

## Libraries Used
```python
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Key Learnings
- Ensemble methods reduce overfitting
- SVM effective in high-dimensional spaces
- Kernel trick for non-linear classification
- Feature importance in Random Forest
- Model comparison techniques