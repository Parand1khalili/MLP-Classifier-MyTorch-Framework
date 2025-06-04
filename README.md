# MLP Classifier & MyTorch Framework

This repository contains two main parts:

###  1. MLP Classifier using PyTorch
A multi-layer perceptron (MLP) was implemented in PyTorch to classify the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset into benign or malignant.  
The model achieves:
-  **96%+** training accuracy  
-  **85%+** test accuracy  

Key Features:
- Data preprocessing with `pandas` and `sklearn`
- Batch training
- Loss/accuracy plots

---

###  2. MyTorch — A Custom Neural Network Library
Reimplemented core deep learning components **from scratch** in Python using `NumPy`.  
MyTorch includes:

- `Tensor` class with basic matrix operations
- `Model` abstraction
- `Linear` (fully connected) layers
- Activation functions: `ReLU`, `Sigmoid`, `LeakyReLU`, `Softmax`
- Loss functions: `MSE`, `CrossEntropy`
- Optimizer: `SGD`


