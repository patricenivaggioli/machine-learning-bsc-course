# Chapter 7 — Ensemble Methods: Random Forests and Boosting

> **Learning objectives:** Understand why combining models works, learn how bagging and Random Forests reduce overfitting, get an intuition for boosting, and know when to use each method.

---

## 7.1 The Wisdom of Crowds: Why Combine Models?

A single decision tree is easy to understand but **unstable and often inaccurate**. What if we trained **many trees** and let them **vote**?

This is the core idea behind **ensemble methods**: combine multiple weak models into one strong model.

```mermaid
flowchart LR
    X["Input data"] --> T1["Tree 1<br/>→ Cat"]
    X --> T2["Tree 2<br/>→ Dog"]
    X --> T3["Tree 3<br/>→ Cat"]
    X --> T4["Tree 4<br/>→ Cat"]
    X --> T5["Tree 5<br/>→ Dog"]
    T1 --> V["Majority Vote<br/>→ Cat (3 vs 2)"]
    T2 --> V
    T3 --> V
    T4 --> V
    T5 --> V

    style X fill:#74b9ff,stroke:#0984e3,color:#000
    style T1 fill:#dfe6e9,stroke:#636e72,color:#000
    style T2 fill:#dfe6e9,stroke:#636e72,color:#000
    style T3 fill:#dfe6e9,stroke:#636e72,color:#000
    style T4 fill:#dfe6e9,stroke:#636e72,color:#000
    style T5 fill:#dfe6e9,stroke:#636e72,color:#000
    style V fill:#55efc4,stroke:#00b894,color:#000
```

**Analogy:** Ask 100 people to guess the number of jellybeans in a jar. Individual guesses vary wildly, but the **average** is surprisingly close to the truth. This is ensemble learning.

---

## 7.2 Bagging and Random Forests

### Bagging (Bootstrap Aggregating)

1. Create many **random subsets** of the training data (with replacement — some samples repeated, some left out)
2. Train one decision tree on each subset
3. Combine predictions: **majority vote** (classification) or **average** (regression)

```mermaid
flowchart TD
    D["Original Dataset<br/>1000 samples"] --> B1["Bootstrap 1<br/>1000 samples<br/>(with repeats)"]
    D --> B2["Bootstrap 2<br/>1000 samples<br/>(with repeats)"]
    D --> B3["Bootstrap 3<br/>1000 samples<br/>(with repeats)"]
    D --> BN["..."]
    B1 --> T1["Tree 1"]
    B2 --> T2["Tree 2"]
    B3 --> T3["Tree 3"]
    BN --> TN["Tree N"]
    T1 --> AGG["Aggregate<br/>(vote or average)"]
    T2 --> AGG
    T3 --> AGG
    TN --> AGG
    AGG --> P["Final Prediction"]

    style D fill:#74b9ff,stroke:#0984e3,color:#000
    style B1 fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style B2 fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style B3 fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style BN fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style T1 fill:#dfe6e9,stroke:#636e72,color:#000
    style T2 fill:#dfe6e9,stroke:#636e72,color:#000
    style T3 fill:#dfe6e9,stroke:#636e72,color:#000
    style TN fill:#dfe6e9,stroke:#636e72,color:#000
    style AGG fill:#a29bfe,stroke:#6c5ce7,color:#000
    style P fill:#55efc4,stroke:#00b894,color:#000
```

### Random Forest = Bagging + Feature Randomness

A Random Forest adds one more trick: at each split, the tree only considers a **random subset of features** (not all of them). This makes the trees more **diverse**, which improves the ensemble.

| Hyperparameter | Meaning | Typical values |
|:---------------|:--------|:---------------|
| `n_estimators` | Number of trees | 100–500 |
| `max_depth` | Maximum tree depth | 5–20 or None |
| `max_features` | Features considered per split | `"sqrt"` (classification), `"log2"` |

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
```

### Why Random Forests work so well

| Single tree problem | How Random Forest fixes it |
|:-------------------|:--------------------------|
| Overfitting | Averaging many trees cancels out noise |
| Instability | Small data changes only affect some trees |
| Low accuracy | Ensemble is more accurate than any single tree |

---

## 7.3 Boosting: Gradient Boosting in a Nutshell

Boosting takes a different approach: instead of training trees **independently**, it trains them **sequentially**, where each new tree focuses on **correcting the mistakes** of the previous ones.

```mermaid
flowchart TD
    D["Training Data"] --> T1["Tree 1<br/>(simple, weak)"]
    T1 --> E1["Errors from Tree 1"]
    E1 --> T2["Tree 2<br/>(focuses on errors)"]
    T2 --> E2["Remaining errors"]
    E2 --> T3["Tree 3<br/>(focuses on remaining errors)"]
    T3 --> EN["...continue..."]
    EN --> F["Final model =<br/>weighted sum of all trees"]

    style D fill:#74b9ff,stroke:#0984e3,color:#000
    style T1 fill:#dfe6e9,stroke:#636e72,color:#000
    style E1 fill:#fab1a0,stroke:#e17055,color:#000
    style T2 fill:#dfe6e9,stroke:#636e72,color:#000
    style E2 fill:#fab1a0,stroke:#e17055,color:#000
    style T3 fill:#dfe6e9,stroke:#636e72,color:#000
    style EN fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style F fill:#55efc4,stroke:#00b894,color:#000
```

### Key idea

- Each tree is **small** (shallow, e.g., depth 3–5) — called a "weak learner"
- Each new tree learns from the **mistakes** of the combined model so far
- The final prediction is the **sum** of all trees (weighted)

### Gradient Boosting in scikit-learn

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
```

| Hyperparameter | Meaning |
|:---------------|:--------|
| `n_estimators` | Number of boosting rounds (trees) |
| `learning_rate` | How much each tree contributes (smaller = slower but more robust) |
| `max_depth` | Depth of each tree (usually 3–5) |

> **Popular libraries:** XGBoost and LightGBM are faster, optimised versions of gradient boosting used in many competitions and real-world applications.

---

## 7.4 When to Use What

```mermaid
flowchart TD
    Q{"What's your priority?"}
    Q -->|"Easy to use<br/>Good default"| RF["Random Forest"]
    Q -->|"Best accuracy<br/>Willing to tune"| GB["Gradient Boosting<br/>(XGBoost / LightGBM)"]
    Q -->|"Interpretability"| DT["Single Decision Tree<br/>(Chapter 6)"]

    RF --> N1["Pros: Fast, robust, hard to mess up<br/>Cons: Slightly less accurate than tuned boosting"]
    GB --> N2["Pros: Often highest accuracy<br/>Cons: More hyperparameters, slower, can overfit"]
    DT --> N3["Pros: Fully interpretable<br/>Cons: Lower accuracy, overfits easily"]

    style Q fill:#ffeaa7,stroke:#fdcb6e,color:#000
    style RF fill:#55efc4,stroke:#00b894,color:#000
    style GB fill:#74b9ff,stroke:#0984e3,color:#000
    style DT fill:#a29bfe,stroke:#6c5ce7,color:#000
    style N1 fill:#dfe6e9,stroke:#636e72,color:#000
    style N2 fill:#dfe6e9,stroke:#636e72,color:#000
    style N3 fill:#dfe6e9,stroke:#636e72,color:#000
```

| Method | Speed | Accuracy | Ease of use | Overfitting risk |
|:-------|:------|:---------|:-----------|:----------------|
| Single tree | Fast | Low–Medium | Very easy | High |
| Random Forest | Fast | High | Easy | Low |
| Gradient Boosting | Medium | Very high | Needs tuning | Medium |

---

## 7.5 Hands-On: Random Forest on a Real Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# --- Load and split ---
X, y = load_wine(return_X_y=True)
target_names = load_wine().target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Compare three models ---
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"{name:20s}  Train: {train_acc:.3f}  Test: {test_acc:.3f}")

# --- Feature importance (Random Forest) ---
rf = models["Random Forest"]
feature_names = load_wine().feature_names

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# --- Effect of number of trees ---
test_scores = []
n_trees_list = [1, 5, 10, 25, 50, 100, 200, 500]
for n in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    test_scores.append(rf.score(X_test, y_test))

plt.figure(figsize=(8, 4))
plt.plot(n_trees_list, test_scores, "o-")
plt.xlabel("Number of Trees")
plt.ylabel("Test Accuracy")
plt.title("Random Forest: Accuracy vs. Number of Trees")
plt.tight_layout()
plt.show()
```

**What you'll see:**
- Random Forest and Gradient Boosting both outperform the single tree
- Accuracy improves quickly with more trees, then plateaus
- Feature importance reveals which inputs matter most (useful for understanding your data)

---

## Summary

```mermaid
mindmap
  root((Chapter 7<br/>Recap))
    Ensemble = many models combined
      Wisdom of crowds
      Vote or average predictions
    Random Forest
      Bagging + feature randomness
      Many independent trees
      Robust, hard to overfit
    Gradient Boosting
      Sequential trees
      Each fixes previous errors
      Often highest accuracy
    Choosing a method
      Start with Random Forest
      Try Boosting for max accuracy
      Single tree only for interpretability
```

---

## Exercises

1. **Conceptual:** Why does averaging predictions from many trees reduce overfitting compared to a single deep tree?
2. **Bagging:** In a bootstrap sample of 100 drawn from 100 original samples (with replacement), roughly what percentage of original samples will be left out? (Hint: the probability of being left out is $(1 - 1/n)^n$.)
3. **Random Forest vs. Boosting:** Explain in your own words the key difference between how Random Forests and Gradient Boosting build their trees.
4. **Hyperparameter tuning:** You train a Gradient Boosting model with `learning_rate=1.0` and `n_estimators=10`. It overfits. What two changes would you try?
5. **Hands-on:** Train a Random Forest and a Gradient Boosting classifier on the Penguins dataset. Compare their test accuracies and feature importances. Which features are most important for identifying penguin species?
