# Lab 5: Implementation of Artificial Neural Network

## Lab Overview
Implementation of Artificial Neural Networks for classification and regression tasks using TensorFlow/Keras.

## Objectives
- Understand ANN architecture
- Implement feedforward neural networks
- Apply activation functions
- Evaluate model performance

## Tasks Completed

### Q1: Logic Gates with Neural Network
- Implemented AND gate using neural network
- Used 1 hidden layer with 4 neurons
- Achieved 100% accuracy
- Compared predictions with actual outputs

### Q2: Regression Task with Neural Network
- Created dataset y = x² + noise
- Built neural network for regression
- Plotted actual vs predicted results
- Analyzed effect of hidden neurons

### Q3: Activation Function Analysis (XOR Gate)
- Trained networks with sigmoid, tanh, ReLU
- Compared accuracy, loss, convergence speed
- Plotted training loss for each activation
- Discussed suitability of each function

### Q4: Binary Classification (Breast Cancer)
- Loaded Breast Cancer dataset
- Built neural network for tumor classification
- Evaluated test accuracy
- Plotted training/validation curves

### Q5: Multi-Class Classification (Iris)
- Trained network to classify 3 flower species
- Used softmax activation for output
- Evaluated model performance
- Made predictions on new samples

### Q6: Regression (California Housing)
- Predicted house prices using neural network
- Used Mean Absolute Error (MAE) metric
- Plotted training history
- Made price predictions

### Q7: Dropout Regularization (MNIST)
- Prevented overfitting using Dropout layers
- Applied to MNIST digit classification
- Compared performance with/without dropout
- Plotted accuracy and loss curves

## Libraries Used
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_breast_cancer, load_iris, fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

## Key Learnings
- Neural network architecture design
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Backpropagation and gradient descent
- Regularization techniques (Dropout)
- Binary vs multi-class classification
- Regression with neural networks