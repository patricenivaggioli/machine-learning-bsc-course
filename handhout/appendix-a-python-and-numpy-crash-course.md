# Appendix A — Python & NumPy Crash Course

> **Goal:** Get comfortable with the Python tools you'll use throughout this guide — variables, loops, functions, NumPy arrays, Pandas DataFrames, and Matplotlib plots.

---

## A.1 Python Basics

### Variables and types

```python
x = 42              # int
pi = 3.14           # float
name = "Alice"      # str
is_student = True   # bool

print(type(x))      # <class 'int'>
```

### Lists

```python
numbers = [10, 20, 30, 40, 50]
print(numbers[0])    # 10  (zero-indexed)
print(numbers[-1])   # 50  (last element)
print(numbers[1:3])  # [20, 30]  (slicing)
numbers.append(60)   # [10, 20, 30, 40, 50, 60]
print(len(numbers))  # 6
```

### Loops

```python
# For loop
for n in numbers:
    print(n)

# Loop with index
for i, n in enumerate(numbers):
    print(f"Index {i}: {n}")

# List comprehension (compact loop)
squares = [n ** 2 for n in range(5)]  # [0, 1, 4, 9, 16]
```

### Conditionals

```python
x = 15
if x > 10:
    print("big")
elif x > 5:
    print("medium")
else:
    print("small")
```

### Functions

```python
def greet(name, excited=False):
    """Return a greeting string."""
    msg = f"Hello, {name}!"
    if excited:
        msg = msg.upper()
    return msg

print(greet("Alice"))              # Hello, Alice!
print(greet("Bob", excited=True))  # HELLO, BOB!
```

### Dictionaries

```python
student = {"name": "Alice", "age": 20, "grade": "A"}
print(student["name"])    # Alice
student["age"] = 21       # update
student["city"] = "Paris" # add new key

for key, value in student.items():
    print(f"{key}: {value}")
```

---

## A.2 NumPy — Fast Numerical Arrays

NumPy is the foundation of scientific Python. Its `ndarray` is like a Python list but **much faster** and supports element-wise operations.

### Creating arrays

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)   # (5,)
print(a.dtype)   # int64

# Useful constructors
zeros = np.zeros((3, 4))        # 3×4 matrix of zeros
ones = np.ones((2, 3))          # 2×3 matrix of ones
rng = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
lin = np.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]
rand = np.random.randn(3, 3)   # 3×3 random normal
```

### Indexing and slicing

```python
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(m[0, 1])     # 2  (row 0, col 1)
print(m[:, 0])     # [1, 4, 7]  (all rows, col 0)
print(m[1, :])     # [4, 5, 6]  (row 1, all cols)
print(m[:2, 1:])   # [[2, 3], [5, 6]]  (first 2 rows, cols 1+)
```

### Element-wise operations (no loops needed!)

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)    # [11, 22, 33]
print(a * b)    # [10, 40, 90]
print(a ** 2)   # [1, 4, 9]
print(np.sqrt(b))  # [3.16, 4.47, 5.48]
```

### Common operations

```python
data = np.array([4, 7, 2, 8, 5, 1, 9, 3, 6])

print(data.mean())     # 5.0
print(data.std())      # 2.58
print(data.min())      # 1
print(data.max())      # 9
print(data.sum())      # 45
print(np.median(data)) # 5.0
print(np.sort(data))   # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Boolean indexing (filtering)

```python
data = np.array([10, 25, 3, 42, 15])
mask = data > 10
print(mask)           # [False, True, False, True, True]
print(data[mask])     # [25, 42, 15]
print(data[data > 10])  # same thing, shorter
```

---

## A.3 Pandas — DataFrames for Tabular Data

Pandas provides the `DataFrame` — a table with labelled columns, ideal for real-world datasets.

### Creating a DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [20, 22, 21, 23],
    "grade": [85, 92, 78, 95],
})
print(df)
```

### Loading data

```python
df = pd.read_csv("data.csv")     # from CSV
print(df.shape)                   # (rows, columns)
print(df.head())                  # first 5 rows
print(df.info())                  # dtypes, non-null counts
print(df.describe())              # statistics
```

### Selecting data

```python
# Single column
print(df["age"])

# Multiple columns
print(df[["name", "grade"]])

# Rows by condition
print(df[df["grade"] > 80])

# Rows by index
print(df.iloc[0])       # first row
print(df.iloc[1:3])     # rows 1 and 2
```

### Common operations

```python
# Add a column
df["passed"] = df["grade"] >= 80

# Group and aggregate
print(df.groupby("passed")["grade"].mean())

# Sort
print(df.sort_values("grade", ascending=False))

# Handle missing values
print(df.isnull().sum())              # count NaN per column
df["age"].fillna(df["age"].median())  # fill NaN with median
df.dropna()                           # drop rows with any NaN
```

---

## A.4 Matplotlib — Quick Plots

### Line plot

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 7, 11, 16]

plt.figure(figsize=(6, 4))
plt.plot(x, y, "o-", label="data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Line Plot")
plt.legend()
plt.tight_layout()
plt.show()
```

### Scatter plot

```python
import numpy as np

x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

plt.figure(figsize=(6, 4))
plt.scatter(x, y, alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot")
plt.tight_layout()
plt.show()
```

### Histogram

```python
data = np.random.randn(1000)

plt.figure(figsize=(6, 4))
plt.hist(data, bins=30, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Histogram")
plt.tight_layout()
plt.show()
```

### Subplots

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot([1, 2, 3], [1, 4, 9])
ax1.set_title("Plot 1")

ax2.bar(["A", "B", "C"], [3, 7, 5])
ax2.set_title("Plot 2")

plt.tight_layout()
plt.show()
```

---

## A.5 Quick Reference Table

| Task | Code |
|:-----|:-----|
| Import NumPy | `import numpy as np` |
| Import Pandas | `import pandas as pd` |
| Import Matplotlib | `import matplotlib.pyplot as plt` |
| Create array | `np.array([1, 2, 3])` |
| Array shape | `a.shape` |
| Mean / std | `a.mean()`, `a.std()` |
| Load CSV | `pd.read_csv("file.csv")` |
| First 5 rows | `df.head()` |
| Filter rows | `df[df["col"] > value]` |
| Group by | `df.groupby("col")["other"].mean()` |
| Plot | `plt.plot(x, y)` then `plt.show()` |
| Scatter | `plt.scatter(x, y)` |
| Histogram | `plt.hist(data, bins=30)` |

---

## Exercises

1. Create a NumPy array of 20 random integers between 1 and 100. Compute the mean, median, and standard deviation.
2. Create a Pandas DataFrame with columns "product", "price", "quantity". Add a computed column "total" = price × quantity. Sort by total descending.
3. Using Matplotlib, plot a histogram of 500 samples from a normal distribution with mean 10 and standard deviation 3.
4. Write a function `describe_array(arr)` that takes a NumPy array and prints its shape, min, max, mean, and std.
5. Load the Iris dataset with scikit-learn (`from sklearn.datasets import load_iris`), put it into a DataFrame, and create a scatter plot of sepal length vs. sepal width coloured by species.
