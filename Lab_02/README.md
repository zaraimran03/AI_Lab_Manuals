## Lab 2 - Multiple & Logistic Regression

**🎯 Objective:**  
Implement multiple linear regression and logistic regression for prediction and classification tasks.

**📚 Topics Covered:**
- Multiple Linear Regression
- Logistic Regression
- Model evaluation metrics (R², MSE, accuracy, precision, recall)
- Decision boundaries
- Linear vs Logistic comparison

**🔧 Key Technologies:**
- `scikit-learn` - ML algorithms
- `numpy` - Numerical computations
- `matplotlib` - Visualization

**✅ Tasks Completed:**

**Q1: House Price Prediction**
- Features: Size (sqft), Bedrooms, Age
- Target: House Price
- Model: Multiple Linear Regression
- Prediction: 2000 sqft, 3 bed, 10 year house
- Result: Coefficients interpreted

**Q2: Student Performance**
- Features: Hours Study, Hours Sleep, Attendance %
- Target: Exam Marks
- Metrics: R² Score, MSE
- Visualization: Actual vs Predicted plot

**Q3: Pass/Fail Classification**
- Features: Hours Study, Hours Sleep
- Target: Pass (1) / Fail (0)
- Model: Logistic Regression
- Visualization: Decision boundary plot
- Prediction: Probability for 30 hours study, 6 hours sleep

**Q4: Diabetes Prediction**
- Features: BMI, Age, Glucose Level
- Target: Diabetic (1) / Not (0)
- Metrics: Accuracy, Precision, Recall
- Prediction: Patient with BMI=28, Age=45, Glucose=150

**Q5: Linear vs Logistic Comparison**
- Dataset: Hours Study, Exam Score, Pass/Fail
- Compared continuous prediction vs probability
- Explained why linear regression fails for classification

**📊 Performance Results:**
```
House Price Prediction:
- R² Score: 0.95+
- MSE: Minimized

Student Performance:
- R² Score: 0.92
- MSE: 45.2

Pass/Fail Classification:
- Accuracy: 95%

Diabetes Prediction:
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
```

**🎓 Key Learnings:**
- When to use linear vs logistic regression
- Coefficient interpretation
- Decision boundary visualization
- Model evaluation metrics

---