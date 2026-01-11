import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# ==========================================
# PRACTICE QUESTION 1
# DNN vs Shallow ANN (Student Pass/Fail)
# ==========================================

print("\n--- PRACTICE QUESTION 1: DNN vs Shallow ANN ---")

data = {
    'StudyHours': ['Low','High','High','Low','High','Medium','Medium'],
    'Attendance': ['Poor','Good','Poor','Good','Good','Good','Poor'],
    'Result': ['Fail','Pass','Pass','Fail','Pass','Pass','Fail']
}

df = pd.DataFrame(data)

encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df[['StudyHours', 'Attendance']].values
y = df['Result'].values

# Shallow ANN
shallow_model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

shallow_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

shallow_model.fit(X, y, epochs=100, verbose=0)
_, shallow_acc = shallow_model.evaluate(X, y, verbose=0)

# Deep DNN
deep_model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

deep_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

deep_model.fit(X, y, epochs=150, verbose=0)
_, deep_acc = deep_model.evaluate(X, y, verbose=0)

print("Shallow ANN Accuracy:", shallow_acc)
print("Deep DNN Accuracy:", deep_acc)


# ==========================================
# PRACTICE QUESTION 2
# Activation Function Analysis (Iris)
# ==========================================

print("\n--- PRACTICE QUESTION 2: ReLU vs Tanh ---")

iris = load_iris()
X = iris.data
y = to_categorical(iris.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ReLU Model
relu_model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

relu_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

relu_model.fit(X_train, y_train, epochs=100, verbose=0)
_, relu_acc = relu_model.evaluate(X_test, y_test, verbose=0)

# Tanh Model
tanh_model = Sequential([
    Dense(16, activation='tanh', input_shape=(4,)),
    Dense(12, activation='tanh'),
    Dense(8, activation='tanh'),
    Dense(3, activation='softmax')
])

tanh_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tanh_model.fit(X_train, y_train, epochs=100, verbose=0)
_, tanh_acc = tanh_model.evaluate(X_test, y_test, verbose=0)

print("ReLU Accuracy:", relu_acc)
print("Tanh Accuracy:", tanh_acc)


# ==========================================
# PRACTICE QUESTION 3
# Hyperparameter Tuning (MNIST)
# ==========================================

print("\n--- PRACTICE QUESTION 3: MNIST DNN ---")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

mnist_model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

mnist_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mnist_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    verbose=1
)

_, mnist_acc = mnist_model.evaluate(X_test, y_test)
print("MNIST Test Accuracy:", mnist_acc)


# ==========================================
# PRACTICE QUESTION 4
# Overfitting & Regularization
# ==========================================

print("\n--- PRACTICE QUESTION 4: Regularization ---")

# Overfitting Model
overfit_model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

overfit_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

overfit_model.fit(X_train, y_train, epochs=10, verbose=1)

# Regularized Model with Dropout
reg_model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

reg_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reg_model.fit(X_train, y_train, epochs=10, verbose=1)

_, reg_acc = reg_model.evaluate(X_test, y_test)
print("Regularized Model Test Accuracy:", reg_acc)


print("\n--- LAB 6 COMPLETED SUCCESSFULLY ---")
