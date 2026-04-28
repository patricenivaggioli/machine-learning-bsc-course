# Appendix C — Notation Cheat Sheet

> **One-page reference** for all the mathematical symbols used in this guide.

---

## Data & Dimensions

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| $n$ | Number of samples (rows) | $n = 150$ |
| $p$ | Number of features (columns) | $p = 4$ |
| $k$ | Number of classes or clusters | $k = 3$ |
| $\mathbf{X}$ | Feature matrix ($n \times p$) | All input data |
| $\mathbf{x}_i$ | Feature vector of sample $i$ | One row of $\mathbf{X}$ |
| $x_{ij}$ | Feature $j$ of sample $i$ | A single cell |
| $\mathbf{y}$ | Target vector | Labels or values to predict |
| $y_i$ | True target for sample $i$ | |

---

## Model Parameters

| Symbol | Meaning | Context |
|:-------|:--------|:--------|
| $\mathbf{w}$ | Weight vector | Linear / logistic regression, NNs |
| $w_j$ | Weight for feature $j$ | |
| $b$ | Bias (intercept) | $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$ |
| $\hat{y}_i$ | Predicted value for sample $i$ | |
| $\hat{y}$ | Prediction (general) | |
| $\theta$ | Generic parameter(s) | Any model |

---

## Functions & Operators

| Symbol | Meaning | Formula / Note |
|:-------|:--------|:--------------|
| $\sum$ | Summation | $\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n$ |
| $\prod$ | Product | $\prod_{i=1}^{n} x_i = x_1 \times x_2 \times \dots \times x_n$ |
| $\bar{x}$ | Mean of $x$ | $\bar{x} = \frac{1}{n}\sum x_i$ |
| $\sigma$ | Standard deviation | $\sigma = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$ |
| $\sigma^2$ | Variance | $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ |
| $\|\mathbf{a}\|$ | Euclidean norm (length) | $\sqrt{\sum a_i^2}$ |
| $\nabla$ | Gradient (vector of derivatives) | $\nabla_\mathbf{w} L$ = gradient of loss w.r.t. weights |
| $\frac{\partial}{\partial w}$ | Partial derivative | Derivative with respect to one variable |
| $\log$ | Natural logarithm ($\ln$) | $\log(e) = 1$ |
| $e^x$ or $\exp(x)$ | Exponential function | $e \approx 2.718$ |
| $\arg\max$ | Value that maximises | $\arg\max_k \, p_k$ = class with highest probability |
| $\arg\min$ | Value that minimises | $\arg\min_\theta \, L(\theta)$ = best parameters |

---

## Activation Functions

| Symbol | Formula | Range |
|:-------|:--------|:------|
| $\sigma(z)$ | $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ |
| $\text{ReLU}(z)$ | $\max(0, z)$ | $[0, \infty)$ |
| $\text{softmax}(z_k)$ | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $(0, 1)$, sums to 1 |

---

## Loss Functions

| Name | Formula | Used for |
|:-----|:--------|:---------|
| MSE | $\frac{1}{n}\sum(\hat{y}_i - y_i)^2$ | Regression |
| MAE | $\frac{1}{n}\sum\lvert\hat{y}_i - y_i\rvert$ | Regression |
| Binary cross-entropy | $-\frac{1}{n}\sum[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ | Binary classification |
| Cross-entropy | $-\frac{1}{n}\sum\sum y_{ik}\log(\hat{y}_{ik})$ | Multi-class |

---

## Evaluation Metrics

| Symbol | Formula | Range |
|:-------|:--------|:------|
| Accuracy | $\frac{\text{correct predictions}}{n}$ | [0, 1] |
| Precision | $\frac{TP}{TP + FP}$ | [0, 1] |
| Recall | $\frac{TP}{TP + FN}$ | [0, 1] |
| F1 | $2 \cdot \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}}$ | [0, 1] |
| $R^2$ | $1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(y_i - \bar{y})^2}$ | $(-\infty, 1]$ |

---

## Abbreviations

| Abbreviation | Full form |
|:-------------|:----------|
| ML | Machine Learning |
| NN | Neural Network |
| MLP | Multi-Layer Perceptron |
| CNN | Convolutional Neural Network |
| PCA | Principal Component Analysis |
| t-SNE | t-distributed Stochastic Neighbour Embedding |
| k-NN | k-Nearest Neighbours |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| TP / FP / TN / FN | True/False Positive/Negative |
| EDA | Exploratory Data Analysis |
| GDPR | General Data Protection Regulation |
| lr | Learning rate |
