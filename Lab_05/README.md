## Lab 5 - Artificial Neural Networks

**🎯 Objective:**  
Build and train feedforward artificial neural networks for various tasks including logic gates, regression, and classification.

**📚 Topics Covered:**
- Feedforward Neural Networks
- Activation functions (sigmoid, tanh, ReLU)
- Binary and multi-class classification
- Regression with neural networks
- Dropout regularization
- Training curves analysis

**🔧 Key Technologies:**
- `tensorflow.keras` - Neural network framework
- `keras.models.Sequential` - Model building
- `keras.layers` - Dense, Dropout

**✅ Tasks Completed:**

**Q1: AND Gate**
- Input: (0,0), (0,1), (1,0), (1,1)
- Output: 0, 0, 0, 1
- Architecture: 4 neurons (ReLU) → 1 output (sigmoid)
- Epochs: 500
- Accuracy: **100%**

**Q2: Regression Task (y = x² + noise)**
- Range: x ∈ [-3, 3]
- Samples: 100
- Architecture: 10 neurons (ReLU) → 1 output
- Loss: MSE
- Visualization: Actual vs Predicted curve

**Q3: XOR Gate (Activation Comparison)**
- Compared: sigmoid, tanh, ReLU
- Best performer: ReLU (fastest convergence)
- Epochs: 500
- Loss and accuracy plots for each

**Q4: Breast Cancer Classification**
- Dataset: 569 samples, 30 features
- Architecture: 16 (ReLU) → 8 (ReLU) → 1 (sigmoid)
- Accuracy: **98%+**
- Epochs: 100
- Visualization: Training/validation curves

**Q5: Iris Multi-Class Classification**
- Classes: Setosa, Versicolor, Virginica
- Architecture: 8 (ReLU) → 6 (ReLU) → 3 (softmax)
- Accuracy: **97%+**
- Loss: Categorical crossentropy

**Q6: California Housing (Regression)**
- Features: 8 (location, rooms, etc.)
- Target: Median house value
- Architecture: 64 (ReLU) → 32 (ReLU) → 1
- Metrics: MSE, MAE
- Epochs: 100

**Q7: MNIST with Dropout**
- Dataset: 60,000 train, 10,000 test
- Architecture: 512 (ReLU) → Dropout(0.3) → 256 (ReLU) → Dropout(0.3) → 10 (softmax)
- Accuracy: **98%+**
- Prevented overfitting successfully

**📊 Performance Results:**
```
Task                    Accuracy/Performance
─────────────────────────────────────────────
AND Gate                100%
XOR (ReLU)              100%
Breast Cancer           98.24%
Iris Classification     97.78%
California Housing      MAE: 0.45
MNIST with Dropout      98.2%
```

**📈 Key Insights:**
- ReLU activation performs best for hidden layers
- Dropout (0.3) effectively prevents overfitting
- Deeper networks needed for complex tasks
- Batch normalization improves training stability

**🎓 Key Learnings:**
- Neural network architecture design
- Activation function selection
- Regularization techniques
- Loss function selection for different tasks

---