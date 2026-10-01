import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from FLOWERS.fuga import fuga_flowers
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


DATA_DIR = Path(__file__).parent.parent.parent / "data"
LUT_PATH = DATA_DIR / "LUTs_Zeta0=0.00e+00_8_16_D80_zhub70_zi400_z0=0.00010000_z70.0_UL_nx2048_ny512_dx20_dy5.NC"
FLUT_PATH = DATA_DIR / "fLUTs_Zeta0=0.00e+00_8_16_D80_zhub70_zi400_z0=0.00010000_z70.0_UL.nc"

# Simple 3-turbine inline layout used across all fuga tests
X = np.array([0., 500., 1000.])
Y = np.array([0., 0., 0.])


def _generic_site():
    """360-direction UniformWeibullSite (1° bins) from wind_rose_10.csv."""
    wind_char = pd.read_csv(DATA_DIR / "wind_rose_10.csv", index_col=0).reset_index()
    return UniformWeibullSite(p_wd=wind_char["P"], a=wind_char["A"],
                              k=wind_char["k"], ti=0.1)


def _nrel_5mw():
    """NREL 5MW reference turbine with tabular Cp and Ct data."""
    diameter = 126.0
    u = np.array([0., 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.,
                  8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.,
                  14.5, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.5, 19., 19.5, 20.,
                  20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.01, 25.02, 50.])
    ct = np.array([0., 0., 0., 0.99, 0.99, 0.97373036, 0.92826162, 0.89210543,
                   0.86100905, 0.835423, 0.81237673, 0.79225789, 0.77584769, 0.7629228,
                   0.76156073, 0.76261984, 0.76169723, 0.75232027, 0.74026851, 0.72987175,
                   0.70701647, 0.54054532, 0.45509459, 0.39343381, 0.34250785, 0.30487242,
                   0.27164979, 0.24361964, 0.21973831, 0.19918151, 0.18131868, 0.16537679,
                   0.15103727, 0.13998636, 0.1289037, 0.11970413, 0.11087113, 0.10339901,
                   0.09617888, 0.09009926, 0.08395078, 0.0791188, 0.07448356, 0.07050731,
                   0.06684119, 0.06345518, 0.06032267, 0.05741999, 0.05472609, 0., 0.])
    cp = np.array([0., 0., 0., 0.178085, 0.289075, 0.349022, 0.384728, 0.406059,
                   0.420228, 0.428823, 0.433873, 0.436223, 0.436845, 0.436575, 0.436511,
                   0.436561, 0.436517, 0.435903, 0.434673, 0.43323, 0.430466, 0.378869,
                   0.335199, 0.297991, 0.266092, 0.238588, 0.214748, 0.193981, 0.175808,
                   0.159835, 0.145741, 0.133256, 0.122157, 0.112257, 0.103399, 0.095449,
                   0.088294, 0.081836, 0.075993, 0.070692, 0.065875, 0.061484, 0.057476,
                   0.053809, 0.050447, 0.047358, 0.044518, 0.0419, 0.039483, 0., 0.])
    p = 0.5 * 1.225 * (diameter ** 2 / 4) * np.pi * u ** 3 * cp / 1e6
    return WindTurbine(name="NREL_5MW", diameter=diameter, hub_height=90,
                       powerCtFunction=PowerCtTabular(ws=u, power=p, power_unit="MW", ct=ct))


# ---------------------------------------------------------------------------
# Module-level fixtures: scope="module" so each model is built once per
# test session, amortising the ~7 s Hankel-transform cost of fLUT init.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fuga_lut_linear():
    """fuga_flowers(source='lut', method='linear') with a 360-direction wind rose."""
    model = fuga_flowers(
        site=_generic_site(), windTurbines=_nrel_5mw(),
        n_terms=10, source="lut", method="linear", lut_file=LUT_PATH,
    )
    return model, X, Y


@pytest.fixture(scope="module")
def fuga_flut_linear():
    """fuga_flowers(source='flut', method='linear') with a 360-direction wind rose.
    The fLUT Hankel transform runs once per module; the result is reused by all tests."""
    model = fuga_flowers(
        site=_generic_site(), windTurbines=_nrel_5mw(),
        n_terms=10, source="flut", method="linear", lut_file=FLUT_PATH,
    )
    return model, X, Y


@pytest.fixture(scope="module")
def fuga_lut_binomial():
    """fuga_flowers(source='lut', method='binomial') with Hornsrev1Site (12 directions).
    A 12-direction wind rose is used because the binomial expansion can give non-physical
    negative AEP with fine-resolution (360-direction) wind roses."""
    model = fuga_flowers(
        site=Hornsrev1Site(), windTurbines=V80(),
        n_terms=6, source="lut", method="binomial", lut_file=LUT_PATH,
    )
    return model, X, Y


@pytest.fixture(scope="module")
def fuga_flut_binomial():
    """fuga_flowers(source='flut', method='binomial') with Hornsrev1Site (12 directions).
    See fuga_lut_binomial for why a 12-direction wind rose is used."""
    model = fuga_flowers(
        site=Hornsrev1Site(), windTurbines=V80(),
        n_terms=6, source="flut", method="binomial", lut_file=FLUT_PATH,
    )
    return model, X, Y


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestFugaLUTLinear:

    """Test suite for fuga_flowers with source='lut' and method='linear' (default configuration)."""

    def test_fuga_lut_linear_creation(self, fuga_lut_linear):
        """
        Test fuga LUT-linear model instantiation.
        Check that the model is created, has the aep_i method, and n_terms is set correctly.
        """
        model, x, y = fuga_lut_linear
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 10


    def test_fuga_lut_linear_aep(self, fuga_lut_linear):
        """
        Test AEP computation for fuga LUT-linear.
        Verify that the result is a positive float.
        """
        model, x, y = fuga_lut_linear

        aep = model.aep(x, y)

        assert isinstance(aep, float)
        assert aep > 0


    def test_fuga_lut_linear_aep_i(self, fuga_lut_linear):
        """
        Test fuga LUT-linear per-turbine AEP.
        Check array length matches the number of turbines and all values are non-negative.
        """
        model, x, y = fuga_lut_linear

        aep_i = model.aep_i(x, y)

        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)


    def test_fuga_lut_linear_delta_p(self, fuga_lut_linear):
        """
        Test fuga LUT-linear _calculate_delta_p method.
        Wake deficits should be a non-negative array of length n_turbines.
        """
        model, x, y = fuga_lut_linear

        delta_p = model._calculate_delta_p(x, y)

        assert len(delta_p) == len(x)
        assert all(delta_p >= 0)


    def test_fuga_lut_linear_p_hat(self, fuga_lut_linear):
        """
        Test fuga LUT-linear p_hat attribute.
        The free-stream power proxy should be a positive float.
        """
        model, x, y = fuga_lut_linear

        p_hat = model.p_hat

        assert isinstance(p_hat, float)
        assert p_hat > 0


    def test_fuga_lut_linear_gradient_autograd_x(self, fuga_lut_linear):
        """
        Test fuga LUT-linear AEP gradient with respect to x using autograd.
        Gradient should be non-None and match the shape of x.
        """
        model, x, y = fuga_lut_linear

        grad_x = model.aep_gradient(gradient_method="Autograd", wrt_arg="x", x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_lut_linear_gradient_autograd_y(self, fuga_lut_linear):
        """
        Test fuga LUT-linear AEP gradient with respect to y using autograd.
        Gradient should be non-None and match the shape of y.
        """
        model, x, y = fuga_lut_linear

        grad_y = model.aep_gradient(gradient_method="Autograd", wrt_arg="y", x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_lut_linear_gradient_autograd_both(self, fuga_lut_linear):
        """
        Test fuga LUT-linear AEP gradient with respect to both x and y using autograd.
        Gradients should be returned as a tuple with correct shapes.
        """
        model, x, y = fuga_lut_linear

        grad_xy = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_lut_linear_gradient_exact_x(self, fuga_lut_linear):
        """
        Test fuga LUT-linear closed-form AEP gradient with respect to x.
        Gradient should be non-None and match the shape of x.
        """
        model, x, y = fuga_lut_linear

        grad_x = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_lut_linear_gradient_exact_y(self, fuga_lut_linear):
        """
        Test fuga LUT-linear closed-form AEP gradient with respect to y.
        Gradient should be non-None and match the shape of y.
        """
        model, x, y = fuga_lut_linear

        grad_y = model.aep_gradient(gradient_method="Exact", wrt_arg=["y"], x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_lut_linear_gradient_exact_both(self, fuga_lut_linear):
        """
        Test fuga LUT-linear closed-form AEP gradient with respect to both x and y.
        Gradients should be returned as a tuple with correct shapes.
        """
        model, x, y = fuga_lut_linear

        grad_xy = model.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_lut_linear_gradient_function(self, fuga_lut_linear):
        """
        Test that fuga LUT-linear returns a callable gradient function when x and y are not provided.
        """
        model, x, y = fuga_lut_linear

        grad_func = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x"])
        grad_x = grad_func(x, y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_lut_linear_plot(self, fuga_lut_linear):
        """
        Test fuga LUT-linear plot_AEP_per_turbine method.
        The method should execute without raising an exception.
        """
        model, x, y = fuga_lut_linear

        try:
            model.plot_AEP_per_turbine(x, y)
            plt.close('all')
        except Exception as e:
            pytest.fail(f"plot_AEP_per_turbine raised {type(e).__name__}: {e}")


    def test_fuga_lut_linear_input_as_list(self, fuga_lut_linear):
        """
        Test that fuga LUT-linear accepts Python lists as x and y inputs.
        """
        model, x, y = fuga_lut_linear

        aep = model.aep(x.tolist(), y.tolist())

        assert aep > 0


class TestFugaFLUTLinear:

    """Test suite for fuga_flowers with source='flut' and method='linear'."""

    def test_fuga_flut_linear_creation(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear model instantiation.
        Check that the model is created, has the aep_i method, and n_terms is set correctly.
        """
        model, x, y = fuga_flut_linear
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 10


    def test_fuga_flut_linear_aep(self, fuga_flut_linear):
        """
        Test AEP computation for fuga fLUT-linear.
        Verify that the result is a positive float.
        """
        model, x, y = fuga_flut_linear

        aep = model.aep(x, y)

        assert isinstance(aep, float)
        assert aep > 0


    def test_fuga_flut_linear_aep_i(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear per-turbine AEP.
        Check array length matches the number of turbines and all values are non-negative.
        """
        model, x, y = fuga_flut_linear

        aep_i = model.aep_i(x, y)

        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)


    def test_fuga_flut_linear_delta_p(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear _calculate_delta_p method.
        Wake deficits should be a non-negative array of length n_turbines.
        """
        model, x, y = fuga_flut_linear

        delta_p = model._calculate_delta_p(x, y)

        assert len(delta_p) == len(x)
        assert all(delta_p >= 0)


    def test_fuga_flut_linear_p_hat(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear p_hat attribute.
        The free-stream power proxy should be a positive float.
        """
        model, x, y = fuga_flut_linear

        p_hat = model.p_hat

        assert isinstance(p_hat, float)
        assert p_hat > 0


    def test_fuga_flut_linear_gradient_autograd_x(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear AEP gradient with respect to x using autograd.
        Gradient should be non-None and match the shape of x.
        """
        model, x, y = fuga_flut_linear

        grad_x = model.aep_gradient(gradient_method="Autograd", wrt_arg="x", x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_flut_linear_gradient_autograd_y(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear AEP gradient with respect to y using autograd.
        Gradient should be non-None and match the shape of y.
        """
        model, x, y = fuga_flut_linear

        grad_y = model.aep_gradient(gradient_method="Autograd", wrt_arg="y", x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_flut_linear_gradient_autograd_both(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear AEP gradient with respect to both x and y using autograd.
        Gradients should be a tuple with correct shapes.
        """
        model, x, y = fuga_flut_linear

        grad_xy = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_flut_linear_gradient_exact_x(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear closed-form AEP gradient with respect to x.
        """
        model, x, y = fuga_flut_linear

        grad_x = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_flut_linear_gradient_exact_y(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear closed-form AEP gradient with respect to y.
        """
        model, x, y = fuga_flut_linear

        grad_y = model.aep_gradient(gradient_method="Exact", wrt_arg=["y"], x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_flut_linear_gradient_exact_both(self, fuga_flut_linear):
        """
        Test fuga fLUT-linear closed-form AEP gradient with respect to both x and y.
        """
        model, x, y = fuga_flut_linear

        grad_xy = model.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_flut_linear_gradient_function(self, fuga_flut_linear):
        """
        Test that fuga fLUT-linear returns a callable gradient function when x and y are not provided.
        """
        model, x, y = fuga_flut_linear

        grad_func = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x"])
        grad_x = grad_func(x, y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


class TestFugaLUTBinomial:

    """Test suite for fuga_flowers with source='lut' and method='binomial'.
    Hornsrev1Site (12 directions) is used to keep the binomial expansion numerically stable."""

    def test_fuga_lut_binomial_creation(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial model instantiation.
        Check model is created, has the aep_i method, and n_terms is set correctly.
        """
        model, x, y = fuga_lut_binomial
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 6


    def test_fuga_lut_binomial_aep(self, fuga_lut_binomial):
        """
        Test AEP computation for fuga LUT-binomial.
        Verify the result is a positive float.
        """
        model, x, y = fuga_lut_binomial

        aep = model.aep(x, y)

        assert isinstance(aep, float)
        assert aep > 0


    def test_fuga_lut_binomial_aep_i(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial per-turbine AEP.
        Check array length matches the number of turbines and all values are non-negative.
        """
        model, x, y = fuga_lut_binomial

        aep_i = model.aep_i(x, y)

        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)


    def test_fuga_lut_binomial_gradient_autograd_x(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial AEP gradient with respect to x using autograd.
        """
        model, x, y = fuga_lut_binomial

        grad_x = model.aep_gradient(gradient_method="Autograd", wrt_arg="x", x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_lut_binomial_gradient_autograd_y(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial AEP gradient with respect to y using autograd.
        """
        model, x, y = fuga_lut_binomial

        grad_y = model.aep_gradient(gradient_method="Autograd", wrt_arg="y", x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_lut_binomial_gradient_autograd_both(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial AEP gradient with respect to both x and y using autograd.
        """
        model, x, y = fuga_lut_binomial

        grad_xy = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_lut_binomial_gradient_exact_x(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial closed-form AEP gradient with respect to x.
        """
        model, x, y = fuga_lut_binomial

        grad_x = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_lut_binomial_gradient_exact_y(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial closed-form AEP gradient with respect to y.
        """
        model, x, y = fuga_lut_binomial

        grad_y = model.aep_gradient(gradient_method="Exact", wrt_arg=["y"], x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_lut_binomial_gradient_exact_both(self, fuga_lut_binomial):
        """
        Test fuga LUT-binomial closed-form AEP gradient with respect to both x and y.
        """
        model, x, y = fuga_lut_binomial

        grad_xy = model.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


class TestFugaFLUTBinomial:

    """Test suite for fuga_flowers with source='flut' and method='binomial'.
    Hornsrev1Site (12 directions) is used to keep the binomial expansion numerically stable."""

    def test_fuga_flut_binomial_creation(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial model instantiation.
        Check model is created, has the aep_i method, and n_terms is set correctly.
        """
        model, x, y = fuga_flut_binomial
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 6


    def test_fuga_flut_binomial_aep(self, fuga_flut_binomial):
        """
        Test AEP computation for fuga fLUT-binomial.
        Verify the result is a positive float.
        """
        model, x, y = fuga_flut_binomial

        aep = model.aep(x, y)

        assert isinstance(aep, float)
        assert aep > 0


    def test_fuga_flut_binomial_aep_i(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial per-turbine AEP.
        Check array length matches the number of turbines and all values are non-negative.
        """
        model, x, y = fuga_flut_binomial

        aep_i = model.aep_i(x, y)

        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)


    def test_fuga_flut_binomial_gradient_autograd_x(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial AEP gradient with respect to x using autograd.
        """
        model, x, y = fuga_flut_binomial

        grad_x = model.aep_gradient(gradient_method="Autograd", wrt_arg="x", x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_flut_binomial_gradient_autograd_y(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial AEP gradient with respect to y using autograd.
        """
        model, x, y = fuga_flut_binomial

        grad_y = model.aep_gradient(gradient_method="Autograd", wrt_arg="y", x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_flut_binomial_gradient_autograd_both(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial AEP gradient with respect to both x and y using autograd.
        """
        model, x, y = fuga_flut_binomial

        grad_xy = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_fuga_flut_binomial_gradient_exact_x(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial closed-form AEP gradient with respect to x.
        """
        model, x, y = fuga_flut_binomial

        grad_x = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_fuga_flut_binomial_gradient_exact_y(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial closed-form AEP gradient with respect to y.
        """
        model, x, y = fuga_flut_binomial

        grad_y = model.aep_gradient(gradient_method="Exact", wrt_arg=["y"], x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_fuga_flut_binomial_gradient_exact_both(self, fuga_flut_binomial):
        """
        Test fuga fLUT-binomial closed-form AEP gradient with respect to both x and y.
        """
        model, x, y = fuga_flut_binomial

        grad_xy = model.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


class TestFugaLUTvsFLUT:

    """Compare LUT and fLUT file sources for accuracy.
    Each comparison uses the same site and turbine so any difference is purely from the file format."""

    def test_lut_vs_flut_linear_error_below_1pct(self, fuga_lut_linear, fuga_flut_linear):
        """
        Test that LUT and fLUT agree within 1% for the linear method.
        Both file formats encode the same Fuga wake shape; any difference is discretisation error.
        """
        model_lut, _, _ = fuga_lut_linear
        model_flut, _, _ = fuga_flut_linear

        aep_lut = model_lut.aep(X, Y)
        aep_flut = model_flut.aep(X, Y)

        relative_error = abs(aep_lut - aep_flut) / aep_lut
        assert relative_error < 0.01, (
            f"LUT-linear vs fLUT-linear relative error {relative_error:.2%} exceeds 1%"
        )


    def test_lut_vs_flut_binomial_error_below_1pct(self, fuga_lut_binomial, fuga_flut_binomial):
        """
        Test that LUT and fLUT agree within 1% for the binomial method.
        Both file formats should produce consistent AEP estimates despite the higher-order expansion.
        """
        model_lut, _, _ = fuga_lut_binomial
        model_flut, _, _ = fuga_flut_binomial

        aep_lut = model_lut.aep(X, Y)
        aep_flut = model_flut.aep(X, Y)

        relative_error = abs(aep_lut - aep_flut) / aep_lut
        assert relative_error < 0.01, (
            f"LUT-binomial vs fLUT-binomial relative error {relative_error:.2%} exceeds 1%"
        )


    def test_fuga_all_configs_produce_different_aep(
        self, fuga_lut_linear, fuga_flut_linear, fuga_lut_binomial, fuga_flut_binomial
    ):
        """
        Test that all four fuga configurations produce distinct AEP values.
        Within each method group (linear/binomial) the two sources should differ;
        across methods, the linear and binomial formulations use different wake expansions.
        """
        lut_lin, _, _ = fuga_lut_linear
        flut_lin, _, _ = fuga_flut_linear
        lut_bin, _, _ = fuga_lut_binomial
        flut_bin, _, _ = fuga_flut_binomial

        aep_lut_lin = lut_lin.aep(X, Y)
        aep_flut_lin = flut_lin.aep(X, Y)
        aep_lut_bin = lut_bin.aep(X, Y)
        aep_flut_bin = flut_bin.aep(X, Y)

        assert aep_lut_lin != aep_flut_lin    # source matters within linear
        assert aep_lut_bin != aep_flut_bin    # source matters within binomial
        assert aep_lut_lin != aep_lut_bin     # method matters within LUT


class TestFugaWindRoseBinSizes:

    """Test that fuga_flowers works correctly across different wind rose bin sizes."""

    def test_fuga_lut_12dir_wind_rose(self):
        """
        Test fuga LUT-linear with a 12-direction wind rose (30° bins) from Hornsrev1Site.
        Checks that AEP is positive and per-turbine array has the right length.
        """
        model = fuga_flowers(
            site=Hornsrev1Site(), windTurbines=V80(),
            n_terms=6, source="lut", method="linear", lut_file=LUT_PATH,
        )

        aep = model.aep(X, Y)
        aep_i = model.aep_i(X, Y)

        assert aep > 0
        assert len(aep_i) == len(X)


    def test_fuga_lut_360dir_wind_rose(self):
        """
        Test fuga LUT-linear with a 360-direction wind rose (1° bins).
        Checks that fine angular resolution is handled correctly.
        """
        model = fuga_flowers(
            site=_generic_site(), windTurbines=_nrel_5mw(),
            n_terms=10, source="lut", method="linear", lut_file=LUT_PATH,
        )

        aep = model.aep(X, Y)

        assert aep > 0


    def test_fuga_bin_size_gives_different_aep(self):
        """
        Test that 12-direction and 360-direction wind roses give different AEP values for fuga.
        Both results must be positive; a difference confirms the model responds to wind rose resolution.
        Uses LUT-linear as the reference configuration (fast init, numerically stable).
        """
        model_12 = fuga_flowers(
            site=Hornsrev1Site(), windTurbines=V80(),
            n_terms=6, source="lut", method="linear", lut_file=LUT_PATH,
        )
        model_360 = fuga_flowers(
            site=_generic_site(), windTurbines=_nrel_5mw(),
            n_terms=10, source="lut", method="linear", lut_file=LUT_PATH,
        )

        aep_12 = model_12.aep(X, Y)
        aep_360 = model_360.aep(X, Y)

        assert aep_12 > 0
        assert aep_360 > 0
        # Different turbines and wind roses give distinct AEP values
        assert aep_12 != aep_360


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
