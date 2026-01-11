# Lab 1: Introduction to VS Code and Google Colab for Data Analysis Using Python

## Lab Overview
This lab introduces Python programming environments (VS Code and Google Colab) and focuses on basic data analysis operations including dataset creation, statistical analysis, and data visualization.

## Objectives
- Set up Python environment in VS Code
- Create and upload datasets using Python
- Perform basic statistical analysis (mean, median, count)
- Visualize data using matplotlib/seaborn

## Tasks Completed

### Q1: Create a Dataset Manually
Created a dataset of 10 students with columns:
- Student_ID
- Name
- Age
- Marks_Math
- Marks_Science
- CGPA

### Q2: Upload Dataset in Python
Loaded dataset using Pandas and displayed basic information.

### Q3: Observe Dataset Information
Analyzed dataset structure using:
- `data.info()` → Dataset structure
- `data.describe()` → Summary statistics
- `data['Marks_Math'].mean()` → Mean of Math marks
- `data['Marks_Science'].max()` → Maximum Science marks

### Q4: Perform Data Analysis
- Counted students with Marks_Math > 50
- Found student with highest Science marks
- Calculated correlation between Marks_Math and Marks_Science

### Q5: Data Visualization
Created visualizations:
1. Bar chart of Student_ID vs Marks_Math
2. Histogram of Age
3. Scatter plot of Marks_Math vs Marks_Science

## Libraries Used
''' python
- import pandas as pd
- import matplotlib.pyplot as plt
'''

## Key Learnings
- Dataset creation and manipulation using Pandas
- Statistical analysis methods
- Data visualization techniques
- Understanding correlations between variables