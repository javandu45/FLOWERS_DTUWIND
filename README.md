# FLOWERS AEP Model

A fast, analytical AEP (Annual Energy Production) computation framework for wind farm optimization.

**FLOWERS** = **FLO**W Estimation and **R**ose **S**uperposition

## Overview

FLOWERS is an efficient AEP estimation model that uses Fourier transform techniques to convert discrete wind direction components into continuous functions, enabling analytical integration and fast gradient computation. This makes it ideal for wind farm layout optimization where many AEP evaluations are needed. The derivations were initially developped by the National Laboratory of the Rockies (NLR) for NO Jensen wake model. This repository contains a FLOWERS implementation for more wake models and is developped by DTU Wind and Energy Systems.

The repository contains a Jupyter Notebook, `examples/AEP_and_gradients.ipynb`, with an explained demonstration of the functionalities. A more detailed demonstration of fuga FLOWERS can be found in `docs/fuga_demo.ipynb`. Its application in WFLO can be seen in `optimization.ipynb`, using DTU's `Topfarm` framwork.

Four wake models are available so far: NO Jensen (`NOJ_flowers`), Gaussian Bastankhah Porte-Agel (`gaussian_flowers`), Nygaard TurbOPark (`TurbOPark_flowers`) and Fuga (`fuga_flowers`), with the later being a work in progress.

The derivation of the TurbOPark FLOWERS can be found in `docs/turbOPark.md` Theoretical background for fuga FLOWERS and its atmoshperic stability implementation can be found in the `docs/fuga.md`.

### Key Features

- **Fast AEP Computation**: Analytical solution instead of numerical integration
- **Gradient Support**: Automatic and exact gradients for optimization
- **Multiple Wake Models**: NO Jensen, Nygaard TurbOPark, Gaussian Bastankhah Porte-Agel and fuga implementations
- **Linear Wake Superposition**: Models wake interactions between turbines
- **PyWAKE Integration**: Compatible with PyWAKE for site/turbine data
- **Atmospheric stability (experimental)**: Fuga FLOWERS has the possibility of considering atmoshperic stability in the AEP estimations.
- **Induction effects**: fuga FLOWERS has the possibility of accounting for turbine induction effects, although limited, in the AEP estimations.

## Installation

### Setup

```bash
# Install via pyproject.toml
pip install -e .
```

## Quick Start

### Basic Usage - NOJ FLOWERS Model

```python
from FLOWERS.noj import NOJ_flowers
from utils import generic_site, nrel_5MW
import numpy as np

# Setup site and turbine
site = generic_site(ws=10)  # Weibull site with avg wind speed 10 m/s
turbine = nrel_5MW()

# Create turbine layout
x = np.array([0, 500, 1000])  # x-positions [m]
y = np.array([0, 500, 1000])  # y-positions [m]

# Initialize FLOWERS model
flowers = NOJ_flowers(site=site, windTurbines=turbine, n_terms=10, k=0.05)

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

# Exact (analytical) gradients
daep_dx, daep_dy = flowers.aep_gradient(
    gradient_method="Exact",
    wrt_arg=["x", "y"],
    x=x, y=y
)
print(f"AEP gradients w.r.t. x: {daep_dx}")
```

### Utility Files

**`examples/utils.py`**
- `nrel_5MW()`: NREL 5MW turbine model (126m diameter)
- `IEA_10MW()`: IEA 10MW turbine model (198m diameter)
- `generic_site(ws)`: Create Weibull wind site from CSV data
- `generate_array()`: Create grid layouts for testing

**`wind_rose_*.csv`**
- Wind characteristic data files
- Columns: `P` (probability), `A` (Weibull scale), `k` (Weibull shape)
- Used by `generic_site()` for different average wind speeds (in this case only 10 m/s is available)

### Test & Demo Files

**`examples/main.py`**
- Demonstrates AEP computation and comparison with PyWAKE
- Shows all FLOWERS models
- Outputs AEP values and computation time

**`examples/original_comparison.py`**
- Detailed comparison between original and integrated versions
- Helps identify numerical differences
- Useful for debugging

**`examples/fuga_comparison.py`**
- Comparison of the different fuga FLOWERS AEP options.
- Setup time, computation time and AEP estimation comparison.
- Optimization comparison with respect to PyWake

**`examples/fuga_demo.ipynb`**
- Explanation of the different setups for fuga FLOWERS.
- AEP and time comparisons.
- Effect of using different number of Fourier terms in the wake reconstruction and AEP estimation.
- Implementation of fuga FLOWERS with atmospheric stability

**`examples/AEP_and_gradients.ipynb`**
- Basic FLOWERS usage.
- AEP and gradients estimation.
- AEP per turbine and plotting.


## Running Examples


```bash
python examples/main.py
```

Outputs AEP comparison between FLOWERS and PyWAKE models, as well as computational times for AEP and its gradients

```bash
python examples/original_comparison.py
```

Compares the AEP output from this FLOWERS package and the original codes written by the authors

```bash
python examples/fuga_comparison.py
```

Compares optimization results of Horns Rev using the different Fuga FLOWERS configurations, and Fuga pywake, all of them evaluated using pywake fuga, to check for advantages of applying fuga with FLOWERS. **Attention**: Pywake fuga can take more than 5 minutes

## Run Tests

```bash
pytest tests/ -v
```

## References

### Academic Papers

1. **NO Jensen FLOWERS**: "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization"
   - DOI: https://doi.org/10.1002/we.2954

2. **Gaussian Bastankhah Porte-Agel FLOWERS**: "Gaussian FLOWERS: Wind-rose-based analytical integration of Gaussian wake model for extremely fast AEP estimation"
   - DOI: https://doi.org/10.1063/5.0245886

3. **Nygaard TurbOPark wake deficit**: "Modelling cluster wakes and wind farm blockage"
   - DOI: https://doi.org/10.1088/1742-6596/1618/6/062072

4. **Fuga wake deficit**: "Developments of the offshore wind turbine wake model Fuga"
   - Link: https://orbit.dtu.dk/en/publications/developments-of-the-offshore-wind-turbine-wake-model-fuga/
