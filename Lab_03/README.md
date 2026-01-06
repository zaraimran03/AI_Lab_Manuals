## Lab 3 - Decision Tree Classifier

**🎯 Objective:**  
Understand decision tree algorithms, entropy, information gain, and implement classification on various datasets.

**📚 Topics Covered:**
- Entropy and Information Gain (manual calculation)
- Decision Tree algorithm
- Tree visualization
- LabelEncoder for categorical data
- Confusion matrix analysis

**🔧 Key Technologies:**
- `scikit-learn` - Decision Tree Classifier
- `seaborn` - Heatmaps
- `matplotlib` - Tree visualization

**✅ Tasks Completed:**

**Q1: Manual Calculation**
- Calculated entropy of Result variable
- Computed information gain for Study Hours
- Determined root node selection

**Q2: Small Dataset Implementation**
- Dataset: Student study data (5 samples)
- Used LabelEncoder for categorical conversion
- Trained DecisionTreeClassifier (criterion='entropy')
- Visualized decision tree
- Predicted: Low study + Good attendance

**Q3: Iris Dataset**
- Split: 70% train, 30% test
- Criterion: Entropy
- Accuracy: **97.78%**
- Identified most important feature at root
- Visualized complete tree structure

**Q4: MNIST Digits Dataset**
- Dataset: 8x8 pixel handwritten digits (1797 samples)
- Classes: 0-9 digits
- Accuracy: **88.33%**
- Confusion matrix heatmap
- Analyzed misclassifications

**📊 Performance Results:**
```
Small Dataset:
- Prediction: Pass (for Low study + Good attendance)

Iris Dataset:
- Test Accuracy: 97.78%
- Most important feature: Petal width

MNIST Digits:
- Test Accuracy: 88.33%
- Best recognized: 0, 6, 7
- Most confused: 1 vs 8
```

**📈 Visualizations:**
- Decision tree structure diagrams
- Confusion matrix heatmaps
- Sample digit images with predictions

**🎓 Key Learnings:**
- Entropy and information gain calculations
- Tree pruning importance
- Handling categorical data
- Decision tree limitations on image data

---
**🎯 Objective:**  
Implement ensemble methods (Random Forest) and Support Vector Machines for classification tasks.

**📚 Topics Covered:**
- Random Forest Classifier
- Support Vector Machines (Linear & RBF kernels)
- Feature importance analysis
- Model comparison
- Confusion matrix interpretation

**🔧 Key Technologies:**
- `scikit-learn.ensemble` - RandomForestClassifier
- `scikit-learn.svm` - SVC
- `pandas` - Data handling

**✅ Tasks Completed:**

**Q1: Iris Classification (Random Forest)**
- Dataset: Iris flower species
- Split: 70-30
- Model: Random Forest (n_estimators=100)
- Accuracy: **100%**

**Q2: Breast Cancer (SVM)**
- Dataset: Breast Cancer (malignant/benign)
- Model: SVM with linear kernel
- Accuracy: **97%+**
- Confusion matrix analysis

**Q3: Student Pass/Fail (Random Forest)**
- Features: study_hours, attendance, marks
- Target: Pass/Fail
- Feature importance computed
- Most important: marks (0.65), study_hours (0.25)

**Q4: Handwritten Digits (SVM RBF)**
- Dataset: Digits (8x8 images)
- Model: SVM with RBF kernel (gamma=0.001, C=10)
- Accuracy: **98%+**
- Visualized misclassified samples

**Q5: Wine Dataset Comparison**
- Compared Random Forest vs SVM (RBF)
- Random Forest: 100% accuracy
- SVM: 97-98% accuracy
- Conclusion: RF performed better on Wine dataset

**📊 Performance Results:**
```
Model Performance Summary:
┌─────────────────────┬──────────────┬──────────┐
│ Dataset             │ Model        │ Accuracy │
├─────────────────────┼──────────────┼──────────┤
│ Iris                │ Random Forest│ 100%     │
│ Breast Cancer       │ SVM (Linear) │ 97%      │
│ Student Pass/Fail   │ Random Forest│ 95%      │
│ Digits              │ SVM (RBF)    │ 98%      │
│ Wine                │ Random Forest│ 100%     │
│ Wine                │ SVM (RBF)    │ 98%      │
└─────────────────────┴──────────────┴──────────┘
```

**📈 Feature Importance (Student Dataset):**
```
marks:       65%
study_hours: 25%
attendance:  10%
```

**🎓 Key Learnings:**
- Ensemble learning advantages
- Kernel selection in SVM
- Feature importance interpretation
- When to use RF vs SVM

---