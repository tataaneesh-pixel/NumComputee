# NumCompute

[![PyPI version](https://badge.fury.io/py/numcompute.svg)](https://badge.fury.io/py/numcompute)
[![Tests](https://github.com/yourusername/numcompute/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/numcompute/actions)
[![Coverage](https://codecov.io/gh/yourusername/numcompute/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/numcompute)

**NumCompute** is a lightweight, NumPy-only scientific computing and machine learning utility toolkit. Production-grade preprocessing, statistics, metrics, optimization, and pipeline tools with zero dependencies beyond NumPy.

## 🚀 Quick Start

```python
pip install numcompute
```

```python
import numpy as np
from numcompute import load_csv, StandardScaler, Pipeline, accuracy

# Load data
X = load_csv("data.csv")

# Build pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
])

X_scaled = pipe.fit_transform(X)

# Evaluate
y_pred = model.predict(X_scaled)
print(f"Accuracy: {accuracy(y_true, y_pred):.3f}")
```

## 📦 Features

- **Data I/O**: CSV loading with missing value handling
- **Preprocessing**: `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`
- **Sorting & Selection**: Stable sort, top-k, quickselect, binary search
- **Statistics**: Descriptive stats, quantiles, histograms (NaN-safe)
- **Ranking**: Average/dense/ordinal ranking + percentiles
- **Metrics**: Accuracy, precision, recall, F1, confusion matrix, MSE, ROC/AUC
- **Optimization**: Finite-difference gradients & Jacobians
- **Pipelines**: Transformer/estimator chaining
- **Utilities**: Stable softmax/logsumexp, distances, activations, batching

**120+ unit tests** • **NumPy-only** • **Production-ready**

## 🛠 Installation

```bash
pip install numcompute
```

**Only requires NumPy** (included automatically).

## 📚 API Overview

### Data Loading
```python
from numcompute import load_csv
X = load_csv("data.csv", delimiter=",", skip_header=True)
```

### Preprocessing Pipeline
```python
from numcompute import Pipeline, StandardScaler, OneHotEncoder

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("encode", OneHotEncoder()),
])
X_clean = pipe.fit_transform(X_raw)
```

### Model Evaluation
```python
from numcompute import accuracy, precision, recall, f1, confusion_matrix

print(f"Accuracy: {accuracy(y_true, y_pred):.3f}")
print(f"F1: {f1(y_true, y_pred):.3f}")
```

### Statistics & Ranking
```python
from numcompute import descriptive_stats, rank, quantile

stats = descriptive_stats(X)
ranks = rank(X, method="average")
q75 = quantile(X, 0.75)
```

## 🎯 Example: End-to-End Workflow

```python
import numpy as np
from numcompute import (
    load_csv, Pipeline, StandardScaler, OneHotEncoder,
    accuracy, confusion_matrix
)

# 1. Load data
X = load_csv("iris.csv")
y = load_csv("iris_labels.csv").flatten()

# 2. Preprocessing pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("encode", OneHotEncoder()),
])
X_processed = pipe.fit_transform(X)

# 3. Train model (your model here)
y_pred = model.predict(X_processed)

# 4. Evaluate
print(f"Accuracy: {accuracy(y, y_pred):.1%}")
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=numcompute --cov-report=html
```

**100% test coverage** across 120+ unit tests.

## 📊 Performance

NumCompute uses pure NumPy vectorization - no Python loops in core computation.

```python
# Benchmark example
from numcompute import StandardScaler
scaler = StandardScaler()
%timeit scaler.fit_transform(X_large)  # Pure NumPy speed
```

## 🛡️ Numerical Stability

- **Stable softmax/logsumexp** with max-shifting
- **NaN-safe** statistics and percentiles  
- **Robust missing value handling**
- **Finite-difference convergence** verified

## 📖 Full Documentation

See the [`docs/` folder](docs/) or build with:

```bash
pip install -e .[docs]
cd docs && make html
```

## 🚀 Development

```bash
git clone https://github.com/yourusername/numcompute
cd numcompute

# Install editable + dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run linting
pre-commit install
pre-commit run --all-files
```

## 🐛 Issues

Found a bug? [Open an issue](https://github.com/yourusername/numcompute/issues/new)!

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

[MIT License](LICENSE) - free to use in commercial and non-commercial projects.

## 🙏 Acknowledgments

Built with ❤️ using NumPy. Thanks to the scientific Python community!

---

⭐ **Star this repo if you found it useful!** ⭐