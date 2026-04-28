# Appendix D — Glossary

> **Alphabetical definitions** of key terms used throughout this guide. Page references indicate the chapter where the term is first introduced or explained in depth.

---

| Term | Definition | Chapter |
|:-----|:----------|:--------|
| **Accuracy** | Fraction of predictions that are correct: $\frac{\text{correct}}{n}$. Can be misleading with imbalanced classes. | 3 |
| **Activation function** | Non-linear function applied to a neuron's output (e.g., ReLU, sigmoid, softmax). | 9 |
| **Backpropagation** | Algorithm that computes gradients of the loss with respect to each weight by working backwards through the network. | 9 |
| **Bagging** | Bootstrap Aggregating — training multiple models on random subsets of data and averaging their predictions. | 7 |
| **Bias (model)** | The intercept term $b$ in a linear model. | 4 |
| **Bias (societal)** | Systematic unfairness in data or predictions that disadvantages certain groups. | 13 |
| **Binary classification** | Predicting one of two classes (e.g., spam / not spam). | 5 |
| **Boosting** | Ensemble technique where models are trained sequentially, each correcting the previous model's errors. | 7 |
| **Categorical feature** | A feature with discrete categories (e.g., colour, country). | 2 |
| **Centroid** | The centre point of a cluster, computed as the mean of all points in the cluster. | 10 |
| **Classification** | Predicting a discrete class label. | 1 |
| **Cluster** | A group of similar data points found by an unsupervised algorithm. | 10 |
| **Confusion matrix** | A table showing counts of true positives, false positives, true negatives, and false negatives. | 3 |
| **Cross-entropy loss** | Loss function for classification: $-\sum y_k \log(\hat{y}_k)$. | 5, 9 |
| **Cross-validation** | Splitting data into $k$ folds and training/evaluating $k$ times to get a robust performance estimate. | 3 |
| **Curse of dimensionality** | The phenomenon where distances become less meaningful and data becomes sparse in high dimensions. | 8 |
| **Data leakage** | When information from the test set accidentally influences training, giving overly optimistic results. | 12 |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise — a clustering algorithm that finds arbitrary-shaped clusters. | 10 |
| **Decision boundary** | The surface in feature space that separates classes. | 5 |
| **Decision tree** | A model that makes predictions by following a series of if/else rules on features. | 6 |
| **Dendrogram** | A tree diagram showing the order and distance of cluster merges in hierarchical clustering. | 10 |
| **Dimensionality reduction** | Reducing the number of features while preserving important information. | 11 |
| **Dot product** | $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ — the core operation in linear models. | B |
| **Elbow method** | Plotting inertia vs. $k$ and looking for the "bend" to choose the number of clusters. | 10 |
| **Ensemble** | A model that combines multiple base models to improve predictions. | 7 |
| **Epoch** | One complete pass through the entire training dataset. | 9 |
| **Euclidean distance** | Straight-line distance: $\|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum(a_i - b_i)^2}$. | 8 |
| **Explained variance ratio** | The fraction of total variance captured by a principal component. | 11 |
| **F1 score** | Harmonic mean of precision and recall: $2 \cdot \frac{P \times R}{P + R}$. | 3 |
| **Feature** | An input variable (column) used by the model to make predictions. | 1, 2 |
| **Feature engineering** | Creating, transforming, or selecting features to improve model performance. | 12 |
| **Gini impurity** | A measure of how mixed the classes are at a tree node; 0 = pure. | 6 |
| **Gradient** | A vector of partial derivatives pointing in the direction of steepest ascent. | 9, B |
| **Gradient descent** | Optimisation algorithm that iteratively moves weights in the direction opposite the gradient to minimise loss. | 9 |
| **Grid search** | Exhaustively trying all combinations of hyperparameter values. | 12 |
| **Hierarchical clustering** | Clustering by progressively merging (or splitting) clusters; produces a dendrogram. | 10 |
| **Hyperparameter** | A model setting not learned from data (e.g., learning rate, $k$ in k-NN, max depth). | 12 |
| **Imputation** | Filling in missing values (e.g., with the mean or median). | 2, 12 |
| **Inertia** | Total within-cluster sum of squared distances to centroids (used in elbow method). | 10 |
| **Information gain** | Reduction in impurity achieved by splitting on a feature. | 6 |
| **Interpretability** | How easily a human can understand why a model made a particular prediction. | 13 |
| **k-fold cross-validation** | Splitting data into $k$ folds; each fold is used once as validation while the rest train. | 3 |
| **k-Means** | Clustering algorithm that partitions data into $k$ spherical clusters by iterating assign-then-update steps. | 10 |
| **k-NN (k-Nearest Neighbours)** | Classifies a point by the majority vote of its $k$ closest neighbours. | 8 |
| **Label** | The known correct answer (target) for a training example. | 1 |
| **Learning rate** | Step size in gradient descent: controls how much weights change per update. | 9 |
| **Linear regression** | A model that predicts a continuous value as a weighted sum of features: $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$. | 4 |
| **Logistic regression** | A classification model that outputs probabilities using the sigmoid function. | 5 |
| **Loss function** | A function that measures how wrong the model's predictions are (lower = better). | 4, 5, 9 |
| **MAE (Mean Absolute Error)** | $\frac{1}{n}\sum\lvert\hat{y}_i - y_i\rvert$ — average absolute prediction error. | 3 |
| **Mean** | Average value: $\bar{x} = \frac{1}{n}\sum x_i$. | B |
| **Min-max scaling** | Scaling features to [0, 1]: $x' = \frac{x - \min}{\max - \min}$. | 2 |
| **MSE (Mean Squared Error)** | $\frac{1}{n}\sum(\hat{y}_i - y_i)^2$ — penalises large errors more. | 3, 4 |
| **Multi-class classification** | Predicting one of three or more classes. | 5 |
| **Multi-Layer Perceptron (MLP)** | A neural network with at least one hidden layer. | 9 |
| **Numerical feature** | A feature with continuous or integer values (e.g., age, income). | 2 |
| **One-hot encoding** | Converting a categorical variable into binary columns (one per category). | 2 |
| **Outlier** | A data point that is very different from the rest. | 2 |
| **Overfitting** | Model performs well on training data but poorly on unseen data (too complex). | 3 |
| **PCA (Principal Component Analysis)** | Dimensionality reduction that finds directions of maximum variance. | 11 |
| **Perceptron** | The simplest neural network: a single neuron with an activation function. | 9 |
| **Pipeline** | A scikit-learn object that chains preprocessing steps and a model into one unit. | 12 |
| **Precision** | Of all predicted positives, how many are actually positive: $\frac{TP}{TP+FP}$. | 3 |
| **Principal component** | A new axis (direction) found by PCA, ranked by variance explained. | 11 |
| **Pruning** | Limiting tree growth (e.g., max_depth) to reduce overfitting. | 6 |
| **$R^2$ (R-squared)** | Fraction of variance explained by the model; 1 = perfect, 0 = predicts the mean. | 3, 4 |
| **Random Forest** | An ensemble of decision trees trained on bootstrap samples with random feature subsets. | 7 |
| **Recall** | Of all actual positives, how many did the model catch: $\frac{TP}{TP+FN}$. | 3 |
| **Regression** | Predicting a continuous numerical value. | 1 |
| **Regularisation** | Techniques that penalise model complexity to reduce overfitting. | 4 |
| **Reinforcement learning** | Learning through trial and error with rewards and penalties. | 1 |
| **ReLU** | Rectified Linear Unit: $\max(0, z)$ — the most common hidden-layer activation. | 9 |
| **RMSE** | Root Mean Squared Error: $\sqrt{MSE}$ — in the original units of the target. | 3 |
| **Sample** | A single data point (row) in the dataset. | 1, 2 |
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ — squashes values to (0, 1). | 5, 9 |
| **Silhouette score** | Measure of how well a point fits its cluster vs. the nearest other cluster; range [−1, 1]. | 10 |
| **Softmax** | Converts a vector of scores into probabilities that sum to 1. | 5, 9 |
| **Standardisation** | Scaling features to mean 0, std 1: $z = \frac{x - \bar{x}}{\sigma}$. | 2 |
| **Supervised learning** | Learning from labelled data (features + known targets). | 1 |
| **t-SNE** | Non-linear dimensionality reduction for visualisation; preserves local structure. | 11 |
| **Target** | The variable the model is trying to predict (also: label, response, dependent variable). | 1, 2 |
| **Test set** | Data held out until the very end for final evaluation. Never used during training or tuning. | 2, 3 |
| **Training set** | Data used to fit the model's parameters. | 2, 3 |
| **Underfitting** | Model is too simple to capture patterns in the data (high bias). | 3 |
| **Unsupervised learning** | Learning from unlabelled data (no targets). | 1, 10 |
| **Validation set** | Data used during development to tune hyperparameters and select models. | 2, 3 |
| **Variance (model)** | Sensitivity of the model to the specific training data used. High variance → overfitting. | 3 |
| **Variance (statistical)** | Measure of data spread: $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$. | B |
| **Weight** | A learnable parameter that scales a feature's contribution to the prediction. | 4, 9 |
