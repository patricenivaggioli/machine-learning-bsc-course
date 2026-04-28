# Appendix B — Math You Actually Need

> **Goal:** Cover just enough maths to understand ML concepts — vectors, basic statistics, probability, gradients, and logarithms. No proofs, just intuition and examples.

---

## B.1 Vectors and Dot Products

A **vector** is just an ordered list of numbers. In ML, a data point with $p$ features is a vector in $\mathbb{R}^p$.

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{pmatrix}$$

### Vector operations

| Operation | Formula | Example |
|:----------|:--------|:--------|
| Addition | $\mathbf{a} + \mathbf{b} = (a_1+b_1, \, a_2+b_2)$ | $(1,2) + (3,4) = (4,6)$ |
| Scalar multiply | $c \cdot \mathbf{a} = (c \cdot a_1, \, c \cdot a_2)$ | $3 \cdot (1,2) = (3,6)$ |
| Dot product | $\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots$ | $(1,2) \cdot (3,4) = 3+8 = 11$ |
| Length (norm) | $\|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2 + \dots}$ | $\|(3,4)\| = \sqrt{9+16} = 5$ |

### Why it matters in ML

- A linear model computes $\mathbf{w} \cdot \mathbf{x} + b$ — that's a dot product!
- **Euclidean distance** between two points: $d = \|\mathbf{a} - \mathbf{b}\|$ (used in k-NN, k-Means)

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))        # 32  (dot product)
print(np.linalg.norm(a))   # 3.74  (length)
print(np.linalg.norm(a - b))  # 5.20  (distance)
```

---

## B.2 Mean, Variance, Standard Deviation

### Mean (average)

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

The "centre" of the data.

### Variance

$$\text{Var}(x) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

How spread out the data is from the mean. Units are squared.

### Standard deviation

$$\sigma = \sqrt{\text{Var}(x)}$$

Same as variance but in the **original units** — easier to interpret.

| Statistic | Formula | Intuition |
|:----------|:--------|:----------|
| Mean $\bar{x}$ | Sum ÷ count | Centre of the data |
| Variance $\sigma^2$ | Average squared distance from mean | Spread (squared units) |
| Std dev $\sigma$ | Square root of variance | Spread (original units) |

```python
import numpy as np

data = np.array([4, 8, 6, 5, 3, 9, 7])
print(f"Mean: {data.mean():.2f}")     # 6.00
print(f"Var:  {data.var():.2f}")      # 3.71
print(f"Std:  {data.std():.2f}")      # 1.93
```

### Why it matters in ML

- **Standardisation** (Chapter 2): $z = \frac{x - \bar{x}}{\sigma}$ → mean 0, std 1
- **Variance** tells PCA which direction to choose (Chapter 11)

---

## B.3 Probability Basics

### What is a probability?

A number between 0 and 1 measuring how likely something is:

$$0 \leq P(A) \leq 1$$

| $P(A)$ | Meaning |
|:-------|:--------|
| 0 | Impossible |
| 0.5 | Equally likely / unlikely |
| 1 | Certain |

### Key rules

| Rule | Formula | Example |
|:-----|:--------|:--------|
| **Complement** | $P(\text{not } A) = 1 - P(A)$ | P(not rain) = 1 − 0.3 = 0.7 |
| **AND** (independent) | $P(A \text{ and } B) = P(A) \times P(B)$ | P(heads AND heads) = 0.5 × 0.5 = 0.25 |
| **OR** (exclusive) | $P(A \text{ or } B) = P(A) + P(B)$ | P(1 or 2 on die) = 1/6 + 1/6 = 1/3 |

### Conditional probability

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

"The probability of A **given that** B happened."

### Why it matters in ML

- **Classification** outputs probabilities: $P(\text{spam} \mid \text{email features})$
- **Logistic regression** (Chapter 5) models $P(y=1 \mid \mathbf{x})$
- **Bayes' theorem** is the foundation of some classifiers

---

## B.4 Gradients (The Slope)

### Derivative = slope of a function

For a function $f(x)$, the derivative $f'(x)$ tells you how fast $f$ is changing:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

| $f'(x)$ | Meaning |
|:---------|:--------|
| Positive | Function is increasing |
| Negative | Function is decreasing |
| Zero | Function is at a minimum or maximum |

### Examples you'll see

| Function | Derivative |
|:---------|:----------|
| $f(x) = x^2$ | $f'(x) = 2x$ |
| $f(x) = 3x + 5$ | $f'(x) = 3$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |

### Gradient = derivative in multiple dimensions

For a function of multiple variables $f(w_1, w_2)$, the **gradient** is a vector of partial derivatives:

$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \end{pmatrix}$$

The gradient **points in the direction of steepest ascent**. To minimise, go in the **opposite direction** — that's gradient descent!

### Why it matters in ML

- **Gradient descent** (Chapters 4, 5, 9): update weights by moving opposite the gradient
- Update rule: $w \leftarrow w - \text{lr} \times \nabla_w L$

---

## B.5 Logarithms

### What is a logarithm?

$\log_b(x) = y$ means $b^y = x$.

In ML, we almost always use the **natural logarithm** $\ln(x) = \log_e(x)$, often written simply as $\log(x)$.

### Key properties

| Property | Formula |
|:---------|:--------|
| Log of 1 | $\log(1) = 0$ |
| Log of product | $\log(ab) = \log(a) + \log(b)$ |
| Log of quotient | $\log(a/b) = \log(a) - \log(b)$ |
| Log of power | $\log(a^n) = n \log(a)$ |

### Why it matters in ML

- **Cross-entropy loss** (logistic regression, neural networks): $L = -\log(p)$
  - If $p = 1$ (correct and confident): $-\log(1) = 0$ (no loss)
  - If $p = 0.01$ (correct but not confident): $-\log(0.01) = 4.6$ (high loss)
- Turns **products of probabilities** into **sums** (numerically more stable)
- **Log transform** can normalise skewed data (Chapter 12)

```python
import numpy as np

p = np.array([0.99, 0.5, 0.01])
print(-np.log(p))   # [0.01, 0.69, 4.61]
# Higher loss when model is less confident about the correct class
```

---

## Quick Reference

| Concept | What to remember |
|:--------|:----------------|
| **Dot product** | $\mathbf{w} \cdot \mathbf{x} = \sum w_i x_i$ — core of linear models |
| **Euclidean distance** | $\|\mathbf{a} - \mathbf{b}\|$ — used in k-NN, k-Means |
| **Mean** | Centre of the data |
| **Standard deviation** | Spread of the data |
| **Probability** | Number in [0, 1] measuring likelihood |
| **Gradient** | Direction of steepest ascent; negate it for gradient descent |
| **Logarithm** | Turns products into sums; used in loss functions |

---

## Exercises

1. Compute the dot product of $\mathbf{a} = (2, 3, 1)$ and $\mathbf{b} = (4, -1, 5)$ by hand. Verify with NumPy.
2. Given data = [2, 4, 4, 4, 5, 5, 7, 9], compute the mean and standard deviation by hand.
3. A fair die is rolled. What is $P(\text{even number})$? What is $P(\text{not } 6)$?
4. If $f(w) = w^2 + 4w$, what is $f'(w)$? At $w = 3$, is the function increasing or decreasing?
5. A model predicts $P(\text{correct class}) = 0.8$. What is the cross-entropy loss $-\log(0.8)$? What if $P = 0.2$?
