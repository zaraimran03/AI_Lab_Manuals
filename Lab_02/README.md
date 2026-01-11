# Lab 2: Implementation of Regression Analysis using Python

## Lab Overview
Implementation of Multiple Linear Regression and Logistic Regression for prediction and classification tasks.

## Objectives
- Understand regression analysis fundamentals
- Implement Multiple Linear Regression
- Implement Logistic Regression
- Evaluate model performance

## Tasks Completed

### Q1: Multiple Linear Regression – House Price Prediction
- Features: Size (sqft), Bedrooms, Age of House
- Target: House Price
- Predicted price for: Size=2000 sqft, Bedrooms=3, Age=10 years
- Interpreted coefficients

### Q2: Multiple Linear Regression – Student Performance
- Features: Hours_Study, Hours_Sleep, Attendance_%
- Target: Marks_in_Exam
- Computed R² score and MSE
- Plotted actual vs predicted marks

### Q3: Logistic Regression – Pass/Fail Classification
- Features: Hours_Study, Hours_Sleep
- Target: Pass (1) / Fail (0)
- Predicted probability of passing
- Plotted decision boundary

### Q4: Logistic Regression – Diabetes Prediction
- Features: BMI, Age, Glucose Level
- Target: Diabetic (1) / Not (0)
- Calculated accuracy, precision, recall
- Made predictions on new patient data

### Q5: Comparison – Linear vs Logistic Regression
- Compared both models on same dataset
- Explained why linear regression is unsuitable for classification

## Libraries Used
''' python
- import pandas as pd
- import numpy as np
- from sklearn.linear_model import LinearRegression, LogisticRegression
- from sklearn.metrics import r2_score, mean_squared_error
- import matplotlib.pyplot as plt
''' 

## Key Learnings
- Difference between regression and classification
- Model evaluation metrics (R², MSE, accuracy, precision, recall)
- Decision boundaries in classification
- When to use linear vs logistic regression