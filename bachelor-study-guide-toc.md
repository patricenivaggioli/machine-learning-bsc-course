# Machine Learning — First-Year Bachelor Study Guide

> **Table of Contents**
> A concise introduction to machine learning fundamentals. No prior ML knowledge required — only basic programming (Python) and high-school mathematics.

---

## Part I — Foundations

### Chapter 1: What Is Machine Learning?
- 1.1 Definitions and Everyday Examples
- 1.2 Why Machine Learning? (vs. Traditional Programming)
- 1.3 Types of Learning: Supervised, Unsupervised, Reinforcement (overview only)
- 1.4 The ML Workflow at a Glance (data → model → prediction)
- 1.5 When ML Works and When It Doesn't

### Chapter 2: Working with Data
- 2.1 What Is a Dataset? (features, labels, samples)
- 2.2 Types of Features: Numerical vs. Categorical
- 2.3 Handling Missing Values and Outliers
- 2.4 Feature Scaling: Why and How (normalization, standardization)
- 2.5 Splitting Data: Train, Validation, Test
- 2.6 Hands-On: Loading and Exploring a Dataset with Pandas

### Chapter 3: How Do We Know a Model Is Good?
- 3.1 Accuracy Is Not Enough
- 3.2 Overfitting vs. Underfitting (intuition with visual examples)
- 3.3 Cross-Validation Made Simple
- 3.4 Key Metrics: Accuracy, Precision, Recall, F1 (classification)
- 3.5 Key Metrics: MSE, MAE, R² (regression)
- 3.6 The Confusion Matrix
- 3.7 Hands-On: Evaluating a Model with scikit-learn

---

## Part II — Supervised Learning

### Chapter 4: Linear Regression
- 4.1 The Idea: Fitting a Line Through Points
- 4.2 Least Squares — How the Line Is Found
- 4.3 Multiple Features (Multiple Linear Regression)
- 4.4 Interpreting Coefficients
- 4.5 When Linear Regression Fails
- 4.6 Hands-On: Predicting House Prices

### Chapter 5: Logistic Regression and Classification
- 5.1 From Regression to Classification
- 5.2 The Sigmoid Function and Decision Boundary
- 5.3 Multi-Class Classification (one-vs-rest)
- 5.4 Interpreting Probabilities
- 5.5 Hands-On: Classifying Iris Species

### Chapter 6: Decision Trees
- 6.1 Intuition: A Flowchart for Decisions
- 6.2 How Splits Are Chosen (Gini Impurity, Information Gain)
- 6.3 Trees for Classification vs. Regression
- 6.4 Overfitting and Pruning
- 6.5 Strengths and Limitations
- 6.6 Hands-On: Visualizing a Decision Tree

### Chapter 7: Ensemble Methods — Random Forests and Boosting
- 7.1 The Wisdom of Crowds: Why Combine Models?
- 7.2 Bagging and Random Forests
- 7.3 Boosting: Gradient Boosting in a Nutshell
- 7.4 When to Use What (decision guide)
- 7.5 Hands-On: Random Forest on a Real Dataset

### Chapter 8: k-Nearest Neighbours (k-NN)
- 8.1 The Simplest Idea: Ask Your Neighbours
- 8.2 Choosing k and Distance Metrics
- 8.3 The Curse of Dimensionality (intuitive explanation)
- 8.4 k-NN for Classification and Regression
- 8.5 Hands-On: Digit Recognition with k-NN

### Chapter 9: Introduction to Neural Networks
- 9.1 Biological Inspiration (very briefly)
- 9.2 The Perceptron: A Single Neuron
- 9.3 Activation Functions (ReLU, sigmoid)
- 9.4 Stacking Layers: The Multi-Layer Perceptron
- 9.5 Training: Gradient Descent and Backpropagation (intuition only)
- 9.6 When to Use Neural Networks (and when not to)
- 9.7 Hands-On: Classifying Handwritten Digits with PyTorch

---

## Part III — Unsupervised Learning

### Chapter 10: Clustering
- 10.1 What Is Clustering? (grouping without labels)
- 10.2 k-Means: Algorithm and Intuition
- 10.3 Choosing k (Elbow Method, Silhouette Score)
- 10.4 Hierarchical Clustering and Dendrograms
- 10.5 A Glimpse at DBSCAN (density-based)
- 10.6 Hands-On: Customer Segmentation

### Chapter 11: Dimensionality Reduction
- 11.1 Why Reduce Dimensions?
- 11.2 PCA: Finding the Most Important Directions
- 11.3 Visualizing High-Dimensional Data (PCA in 2D/3D)
- 11.4 t-SNE for Visualization (intuition, no math)
- 11.5 Hands-On: Visualizing the MNIST Dataset

---

## Part IV — Putting It All Together

### Chapter 12: The Complete ML Pipeline
- 12.1 Problem Definition: Asking the Right Question
- 12.2 Data Collection and Cleaning Checklist
- 12.3 Feature Engineering Basics
- 12.4 Model Selection: Which Algorithm to Try First?
- 12.5 Hyperparameter Tuning (Grid Search, Random Search)
- 12.6 scikit-learn Pipelines
- 12.7 Hands-On: End-to-End Mini-Project (Titanic or similar)

### Chapter 13: Ethics and Limitations
- 13.1 Bias in Data, Bias in Models
- 13.2 Fairness: A Simple Example
- 13.3 Interpretability: Why "Black Box" Matters
- 13.4 Privacy Basics (what data should you not use?)
- 13.5 Responsible ML Checklist

---

## Appendices

### Appendix A — Python and NumPy Crash Course
- A.1 Python Basics Refresher (lists, dicts, loops, functions)
- A.2 NumPy: Arrays and Vectorized Operations
- A.3 Pandas: DataFrames in 10 Minutes
- A.4 Matplotlib: Plotting Essentials

### Appendix B — Math You Actually Need
- B.1 Vectors and Dot Products
- B.2 Means, Variance, Standard Deviation
- B.3 Probability Basics (conditional probability, Bayes' rule)
- B.4 What Is a Gradient? (slope in multiple dimensions)
- B.5 Logarithms and Exponentials (for logistic regression and entropy)

### Appendix C — Notation Cheat Sheet
- One-page table of symbols used in the guide

### Appendix D — Glossary
- Alphabetical definitions of key terms (50–70 entries)

### Appendix E — What to Learn Next
- Pointers to the MSc-level study guide, online courses, and further reading
