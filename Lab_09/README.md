# Lab 9: Convolutional Neural Network (OEL)

## Lab Overview
Open-ended lab to build a CNN for image classification (Cats vs Dogs).

## Objectives
- Collect and organize image dataset
- Build CNN architecture
- Train and evaluate model
- Visualize results

## Tasks Completed

### Dataset Collection
- Downloaded 500 cat images
- Downloaded 500 dog images
- Organized in dataset/cats and dataset/dogs folders

### CNN Implementation
- Loaded and preprocessed images (150×150)
- Split into 80% training, 20% testing
- Built CNN with Conv2D, MaxPooling, Dropout layers
- Trained for 10-20 epochs
- Evaluated accuracy

### Visualization
- Plotted training/validation accuracy
- Plotted training/validation loss
- Created confusion matrix
- Made predictions on test images
- Displayed "This image is a CAT/DOG" results

## Libraries Used
```python
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
```

## Key Learnings
- CNN architecture for image classification
- Convolutional layers extract features
- Pooling layers reduce dimensionality
- Dropout prevents overfitting
- Data preprocessing importance
- Model evaluation techniques