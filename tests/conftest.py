import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids display/GUI dependency in tests

import pytest
import pandas as pd
from pathlib import Path
from py_wake.site import UniformWeibullSite


DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def generic_site():
    """360-direction UniformWeibullSite (1° bins) built from wind_rose_10.csv."""
    wind_char = pd.read_csv(DATA_DIR / "wind_rose_10.csv", index_col=0).reset_index()
    return UniformWeibullSite(
        p_wd=wind_char["P"],
        a=wind_char["A"],
        k=wind_char["k"],
        ti=0.1,
    )
