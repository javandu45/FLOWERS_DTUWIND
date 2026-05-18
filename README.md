# FLOWERS AEP Model

A fast, analytical AEP (Annual Energy Production) computation framework for wind farm optimization.

**FLOWERS** = **FLO**W Estimation and **R**ose **S**uperposition

## Overview

FLOWERS is an efficient AEP estimation model that uses Fourier transform techniques to convert discrete wind direction components into continuous functions, enabling analytical integration and fast gradient computation. This makes it ideal for wind farm layout optimization where many AEP evaluations are needed.

The repository contains a Jupyter Notebook, `demo.ipynb`, with an explained demonstration of the functionalities.

### Key Features

- **Fast AEP Computation**: Analytical solution instead of numerical integration
- **Gradient Support**: Automatic and exact gradients for optimization
- **Multiple Wake Models**: NO Jensen and Gaussian Bastankhah implementations
- **Linear Wake Superposition**: Models wake interactions between turbines
- **PyWAKE Integration**: Compatible with PyWAKE for site/turbine data

## Installation

### Setup

```bash
# Install via pyproject.toml
pip install -e .
```

## Quick Start

### Basic Usage - NOJ FLOWERS Model

```python
from FLOWERS_integrated import NOJ_flowers
from utils import generic_site, nrel_5MW
import numpy as np

# Setup site and turbine
site = generic_site(ws=11)  # Weibull site with avg wind speed 11 m/s
turbine = nrel_5MW()

# Create turbine layout
x = np.array([0, 500, 1000])  # x-positions [m]
y = np.array([0, 500, 1000])  # y-positions [m]

# Initialize FLOWERS model
flowers = NOJ_flowers(site=site, WindTurbine=turbine, n_terms=10, k=0.05)

# Compute AEP
aep = flowers.aep(x, y)
print(f"Farm AEP: {aep:.2f} GWh")
```

### Computing Gradients

```python
# Automatic differentiation gradients
daep_dx = flowers.aep_gradient(
    gradient_method="Autograd", 
    wrt_arg=["x"], 
    x=x, y=y
)

# Exact (analytical) gradients (NOJ only)
daep_dx, daep_dy = flowers.aep_gradient(
    gradient_method="Exact",
    wrt_arg=["x", "y"],
    x=x, y=y
)
print(f"AEP gradients w.r.t. x: {daep_dx}")
```

### Utility Files

**`utils.py`**
- `nrel_5MW()`: NREL 5MW turbine model (126m diameter)
- `IEA_10MW()`: IEA 10MW turbine model (198m diameter)
- `generic_site(ws)`: Create Weibull wind site from CSV data
- `generate_array()`: Create grid layouts for testing

**`wind_rose_*.csv`**
- Wind characteristic data files
- Columns: `P` (probability), `A` (Weibull scale), `k` (Weibull shape)
- Used by `generic_site()` for different average wind speeds (in this case only 10 m/s is available)

### Test & Demo Files

**`demo.py`**
- Demonstrates AEP computation and comparison with PyWAKE
- Shows both NOJ and Gaussian models
- Outputs AEP values and computation time

**`original_comparison.py`**
- Detailed comparison between original and integrated versions
- Helps identify numerical differences
- Useful for debugging


## Running Examples


```bash
python examples/main.py
```

Outputs AEP comparison between FLOWERS and PyWAKE models, as well as computational times for AEP and its gradients

```bash
python examples/original_comparison.py
```

Compares the AEP output from this FLOWERS package and the original codes written by the authors

## Run Tests

```bash
pytest tests/coverage_tests.py
```

## References

### Academic Papers

1. **NO Jensen FLOWERS**: "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization"
   - DOI: https://doi.org/10.1002/we.2954

2. **Gaussian FLOWERS**: "Gaussian FLOWERS: Wind-rose-based analytical integration of Gaussian wake model for extremely fast AEP estimation"
   - DOI: https://doi.org/10.1063/5.0245886
