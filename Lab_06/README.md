# Lab 6: Implementation of Deep Neural Network

## Lab Overview
Implementation of Deep Neural Networks with multiple hidden layers for complex pattern recognition.

## Objectives
- Understand deep learning architectures
- Implement DNNs with multiple layers
- Analyze hyperparameter effects
- Apply regularization techniques

## Tasks Completed

### Practice Question 1: DNN Architecture Design
- Created DNN with 3+ hidden layers
- Experimented with different neuron counts
- Compared with shallow ANN (1 hidden layer)
- Analyzed accuracy improvements

### Practice Question 2: Activation Function Analysis
- Built two DNNs on Iris dataset
- Model A: ReLU activation
- Model B: tanh activation
- Compared accuracy and training behavior
- Documented observations

### Practice Question 3: Hyperparameter Tuning (MNIST)
- Varied number of hidden layers
- Changed neurons per layer
- Modified batch sizes
- Recorded effects on training time and accuracy

### Practice Question 4: Overfitting and Regularization
- Observed overfitting signs
- Applied Dropout regularization
- Implemented Early Stopping
- Compared before/after regularization

## Libraries Used
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

## Key Learnings
- Deep architectures learn hierarchical features
- More layers ≠ always better (diminishing returns)
- ReLU generally outperforms tanh in deep networks
- Larger batch sizes train faster but may reduce accuracy
- Regularization essential for preventing overfitting
- Early stopping saves computation time