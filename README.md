# 🤖 AI Lab Manuals - Complete Collection

## 📋 Student Information
- **Name:** Zara Imran  
- **Registration No:** 70149236  
- **Program:** BS IET  
- **Department:** Technology

---

## 📚 Repository Overview

This repository contains a comprehensive collection of **14 lab assignments** covering fundamental to advanced concepts in **Artificial Intelligence** and **Machine Learning**. Each lab includes complete implementation, detailed explanations, visualizations, and practical applications.

---

## 🗂️ Repository Structure

```
Lab_01/
├── Assignment1.ipynb
├── LAB_01.pdf
└── README.md

Lab_02/
├── Assignment2.ipynb
├── LAB_02.pdf
├── README.md
├── Student_performance dataset.xlsx
├── diabetes dataset.xlsx
├── exam_comparison dataset.xlsx
└── house_price.csv

Lab_03/
├── Assignment3.ipynb
├── LAB_03.pdf
└── README.md

Lab_04/
├── Assignment4.ipynb
├── LAB_04.pdf
└── README.md

Lab_05/
├── Assignment4.ipynb
├── LAB_05.pdf
└── README.md

Lab_06/
├── Assignment6.ipynb
├── LAB_06.pdf
└── README.md

Lab_07/
├── Assignment7.ipynb
├── LAB_07.pdf
└── README.md

Lab_08/
├── Assignment8.ipynb
├── LAB_08.pdf
└── README.md

Lab_09/
├── Assignment9.ipynb
├── LAB_09.pdf
└── README.md

Lab_10/
├── Assignment10(pt-1).ipynb
├── Assignment10(pt-2).ipynb
├── LAB_10(pt-1).pdf
├── LAB_10(pt-2).pdf
└── README.md

Lab_11/
├── Assignment11.ipynb
├── LAB_11.pdf
├── README.md
└── owid-covid-data.csv

Lab_12/
├── Assignment12.ipynb
├── LAB_12.pdf
└── README.md

Lab_13/
├── 001ssb.txt
├── 002ssb.txt
├── 003ssb.txt
├── 004ssb.txt
├── 005ssb.txt
├── LAB_13.pdf
└── README.md

Lab_14/
├── LAB_14.pdf
└── LangChain_RAG_Lab14/
├── README.md
├── Social_Network_Ads.csv
├── compare_loaders.py
├── cricket.txt
├── csv_loader.py
├── directory_loader.py
├── dl-curriculum.pdf
├── pdf_loader.py
├── text_loader.py
└── webbase_loader.py
```

---

## 📖 Lab Contents

### 🔹 Lab 01: Dataset Creation & Analysis
**Topics:** Pandas, Data Manipulation, Statistical Analysis, Data Visualization
- Manual dataset creation with student records
- CSV file operations
- Statistical analysis (mean, max, correlation)
- Visualizations: Bar charts, Histograms, Scatter plots

**Key Skills:** Data preprocessing, exploratory data analysis, Matplotlib

---

### 🔹 Lab 02: Regression Models
**Topics:** Multiple Linear Regression, Logistic Regression
- House price prediction with multiple features
- Student performance prediction
- Pass/fail classification
- Diabetes prediction
- Model evaluation (R², MSE, Accuracy, Precision, Recall)

**Key Skills:** Regression analysis, classification, model interpretation

---

### 🔹 Lab 03: Decision Trees
**Topics:** Decision Tree Classifier, Entropy, Information Gain
- Manual entropy and information gain calculation
- Tree visualization
- Iris dataset classification (97.78% accuracy)
- MNIST digit recognition (88.33% accuracy)
- Confusion matrix analysis

**Key Skills:** Tree-based models, feature importance, visualization

---

### 🔹 Lab 04: Ensemble Methods & SVM
**Topics:** Random Forest, Support Vector Machines
- Iris flower classification
- Breast cancer detection
- Handwritten digit recognition with SVM
- Feature importance analysis
- Model comparison: Random Forest vs SVM

**Key Skills:** Ensemble learning, kernel methods, hyperparameter tuning

---

### 🔹 Lab 05: Artificial Neural Networks
**Topics:** Feed-forward Networks, Activation Functions, Regularization
- Logic gates implementation (AND, XOR)
- Regression tasks (quadratic function)
- Binary classification (breast cancer - 98.25% accuracy)
- Multi-class classification (Iris)
- House price prediction
- Dropout regularization on MNIST (97.89% accuracy)

**Key Skills:** Neural network architecture, activation functions, overfitting prevention

---

### 🔹 Lab 06: Advanced Neural Networks
**Topics:** Neural Network Optimization
- Advanced architectures
- Hyperparameter tuning
- Performance optimization

**Key Skills:** Model optimization, fine-tuning

---

### 🔹 Lab 07: Recurrent Neural Networks
**Topics:** RNN, Sequential Data Processing
- Next word prediction
- Stock price forecasting
- Sentiment analysis on IMDb reviews
- Weather forecasting
- Music note generation

**Key Skills:** Sequence modeling, time series analysis, text generation

---

### 🔹 Lab 08: Long Short-Term Memory (LSTM)
**Topics:** LSTM Networks, Advanced Text Processing
- Game of Thrones text prediction
- Custom text corpus training
- Multi-word sequential generation
- Long-term dependency handling

**Key Skills:** LSTM architecture, advanced NLP, text generation

---

### 🔹 Lab 09: Convolutional Neural Networks
**Topics:** CNN, Image Classification
- **Open-ended Project:** Cat vs Dog Classifier
- Dataset: 1000 images (500 cats + 500 dogs)
- Image preprocessing and augmentation
- Model training with visualization
- Confusion matrix and prediction

**Key Skills:** Computer vision, image processing, CNN architecture

---

### 🔹 Lab 10: K-Means Clustering
**Topics:** Unsupervised Learning, Clustering
- K-Means from scratch implementation
- Scikit-learn KMeans with K=2,3,4
- Effect of new data points on clusters
- Distance table creation
- Cluster visualization

**Key Skills:** Clustering algorithms, centroid calculation, visualization

---

### 🔹 Lab 11: Fuzzy C-Means Clustering
**Topics:** Soft Clustering, Membership Degrees
- FCM on synthetic 2D data
- Iris dataset clustering (90% accuracy)
- Image segmentation
- Market segmentation analysis
- COVID-19 country clustering
- Comparison with K-Means

**Key Skills:** Fuzzy logic, soft clustering, membership analysis

---

### 🔹 Lab 12: Reinforcement Learning Basics
**Topics:** RL Fundamentals, Agent-Environment Interaction
- CartPole environment experiments
- MountainCar environment
- Rule-based policies
- State-action-reward framework
- Episode tracking and visualization

**Key Skills:** RL fundamentals, Gymnasium, policy design

---

### 🔹 Lab 13: Natural Language Processing
**Topics:** Word2Vec, Word Embeddings
- Word2Vec training on Game of Thrones dataset
- Word similarity analysis
- Character relationship exploration
- 3D visualization with PCA
- Skip-Gram vs CBOW comparison

**Key Skills:** NLP, word embeddings, semantic analysis

---

### 🔹 Lab 14: Document Loading for RAG
**Topics:** LangChain, Retrieval-Augmented Generation
- PyPDFLoader for PDF documents
- CSVLoader for structured data
- WebBaseLoader for web content
- TextLoader for plain text
- Metadata extraction and comparison

**Key Skills:** RAG systems, document processing, LangChain

---

## 🛠️ Technologies & Libraries

### Core Libraries
```python
# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Dropout

# NLP
import nltk
import gensim
from langchain_community.document_loaders import PyPDFLoader

# Clustering
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Reinforcement Learning
import gymnasium as gym
import pygame
```

---

## 📊 Key Achievements

| Lab | Topic | Best Result |
|-----|-------|-------------|
| Lab 03 | Decision Tree on Iris | 97.78% accuracy |
| Lab 04 | Random Forest & SVM | High classification accuracy |
| Lab 05 | ANN on Breast Cancer | 98.25% accuracy |
| Lab 05 | ANN on MNIST | 97.89% accuracy |
| Lab 09 | CNN Cat vs Dog | 90% validation accuracy |
| Lab 11 | FCM on Iris | 90% accuracy with soft clustering |
| Lab 12 | RL MountainCar | Successfully solved with momentum policy |

---

## 🎯 Learning Outcomes

After completing these labs, I have gained expertise in:

### Machine Learning
- ✅ Supervised Learning (Classification & Regression)
- ✅ Unsupervised Learning (Clustering)
- ✅ Ensemble Methods
- ✅ Model Evaluation & Validation

### Deep Learning
- ✅ Artificial Neural Networks
- ✅ Convolutional Neural Networks
- ✅ Recurrent Neural Networks
- ✅ LSTM Networks
- ✅ Regularization Techniques

### Specialized Topics
- ✅ Natural Language Processing
- ✅ Computer Vision
- ✅ Reinforcement Learning
- ✅ Fuzzy Logic
- ✅ RAG Systems

### Practical Skills
- ✅ Data Preprocessing & Visualization
- ✅ Model Training & Optimization
- ✅ Hyperparameter Tuning
- ✅ Performance Evaluation
- ✅ Real-world Problem Solving

---

## 📈 Performance Metrics Summary

### Classification Tasks
- Breast Cancer Detection: **98.25%**
- MNIST Digit Recognition: **97.89%**
- Iris Classification: **97.78%**
- Cat vs Dog Classification: **90%**

### Clustering Tasks
- K-Means: Successfully clustered various datasets
- Fuzzy C-Means: Achieved meaningful soft clustering

### Reinforcement Learning
- CartPole: Rule-based policy significantly outperformed random
- MountainCar: Successfully reached goal with momentum strategy

---

## 🚀 How to Use This Repository

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tensorflow keras
pip install nltk gensim plotly
pip install gymnasium pygame
pip install langchain langchain-community
pip install skfuzzy opencv-python
```

### Running the Labs
1. Clone the repository
2. Navigate to specific lab folder
3. Open Jupyter notebook
4. Run cells sequentially
5. Refer to README.md in each lab for detailed instructions

---

## 📝 Lab Format

Each lab follows a consistent structure:

1. **Objective** - Clear learning goals
2. **Theory** - Conceptual explanation
3. **Implementation** - Step-by-step code
4. **Visualization** - Graphs and plots
5. **Analysis** - Results interpretation
6. **Questions** - Conceptual understanding
7. **Conclusion** - Key takeaways

---

## 🔍 Topics Quick Reference

| Topic | Labs |
|-------|------|
| **Data Analysis** | Lab 01 |
| **Regression** | Lab 02 |
| **Classification** | Lab 02, 03, 04, 05 |
| **Neural Networks** | Lab 05, 06 |
| **RNN/LSTM** | Lab 07, 08 |
| **CNN** | Lab 09 |
| **Clustering** | Lab 10, 11 |
| **Reinforcement Learning** | Lab 12 |
| **NLP** | Lab 13 |
| **RAG Systems** | Lab 14 |

---

## 💡 Highlights

### Most Challenging Labs
- **Lab 09:** CNN Cat vs Dog (open-ended, large dataset)
- **Lab 12:** Reinforcement Learning (complex environment dynamics)
- **Lab 13:** Word2Vec (large corpus processing)

### Most Practical Labs
- **Lab 02:** Regression (real-world prediction tasks)
- **Lab 09:** CNN (image classification application)
- **Lab 11:** Fuzzy C-Means (market segmentation)
- **Lab 14:** RAG (modern AI applications)

### Best Visualizations
- **Lab 10:** K-Means cluster evolution
- **Lab 13:** 3D word embeddings
- **Lab 05:** Training curves and confusion matrices

---

## 🎓 Course Information

**Institution:** University of Lahore  
**Department:** Technology  
**Program:** BS IET  
**Course:** Artificial Intelligence Lab  
**Academic Year:** 2025-2026

---

## 📧 Contact

**Student:** Zara Imran  
**Registration No:** 70149236  
**Email:** zarachoudhry325@gmail.com  
**GitHub:** zaraimran03

---


## 🙏 Acknowledgments

- Course Instructor and Lab Supervisors
- Open-source community for libraries and tools
- Kaggle for datasets
- TensorFlow and Scikit-learn documentation

---

## 🔄 Updates

| Date | Update |
|------|--------|
| Jan 2026 | Initial repository creation |
| Jan 2026 | Completed all 14 labs |
| Jan 2026 | Added comprehensive READMEs |

---

## ⭐ Repository Stats

- **Total Labs:** 14
- **Total Questions:** 50+
- **Lines of Code:** 5000+
- **Datasets Used:** 20+
- **Models Trained:** 30+
- **Accuracy Achieved:** Up to 98.25%

---

<div align="center">

### 🎯 "From Data to Intelligence - A Complete AI Journey"

**Made by Zara Imran**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

</div>
