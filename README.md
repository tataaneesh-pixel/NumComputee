# NumCompute

##  Assignment Overview

This project was developed as part of a Python programming and Artificial Intelligence/Machine Learning coursework assignment.

The goal of this project is to demonstrate understanding and practical implementation of core programming concepts, including:

* Python fundamentals (variables, loops, conditionals, functions)
* Data structures and algorithms
* Object-Oriented Programming (OOP)
* File handling and exception handling
* Functional programming concepts
* Unit testing using `pytest`

In addition, the project applies these concepts to build a small numerical computing and machine learning utility library using **NumPy only**.

---

##  Project Description

**NumCompute** is a lightweight Python toolkit that provides basic utilities for:

* Data loading and preprocessing
* Statistical analysis
* Ranking and sorting
* Machine learning evaluation metrics
* Numerical optimization (gradients and Jacobians)
* Pipeline-based workflows

The project focuses on writing clean, modular, and testable code while handling edge cases such as invalid inputs and missing values.

---

##  Features

* **Data I/O**: CSV loading with optional preprocessing
* **Preprocessing**: Standard scaling, normalization, encoding, and imputation
* **Statistics**: Mean, standard deviation, quantiles, and histograms (NaN-safe)
* **Ranking**: Average, dense, and ordinal ranking methods
* **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
* **Optimization**: Numerical gradient and Jacobian computation
* **Pipelines**: Simple transformation pipelines for data workflows
* **Utilities**: Distance functions, activation functions, batching tools

---

##  Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>
cd NumComputee-main
pip install numpy pytest
```

---

##  How to Run

Run all tests using:

```bash
python -m pytest -v
```

Run a specific test file:

```bash
python -m pytest tests/test_utils.py -v
```

---

##  Testing

This project uses **pytest** for unit testing.

* All modules are tested
* Edge cases are handled (invalid inputs, empty arrays, NaNs)
* Functions are validated against expected outputs

---

##  Project Structure

```
numcompute/
    io.py
    preprocessing.py
    stats.py
    rank.py
    optim.py
    utils.py

tests/
    test_io.py
    test_preprocessing.py
    test_stats.py
    test_rank.py
    test_optim.py
    test_utils.py
```

---

##  Concepts Demonstrated

This project demonstrates the following key concepts:

* Modular programming and code organization
* Reusable functions and classes
* Input validation and error handling
* Numerical computation using NumPy
* Writing testable code and debugging using pytest
* Basic machine learning workflow design

---

##  Example Usage

```python
import numpy as np
from numcompute import descriptive_stats, Pipeline, StandardScaler

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Compute statistics
stats = descriptive_stats(X)
print(stats)

# Apply preprocessing pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
])

X_scaled = pipe.fit_transform(X)
print(X_scaled)
```

---

##  Notes

* This project was developed for educational purposes.
* Only NumPy and standard Python libraries were used.
* The focus is on correctness, clarity, and testing rather than performance optimization.

---

##  License

This project is submitted as part of coursework and is intended for academic use.
