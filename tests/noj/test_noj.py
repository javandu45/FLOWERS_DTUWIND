import pytest
import numpy as np
from matplotlib import pyplot as plt

from FLOWERS.noj import NOJ_flowers
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


class TestNOJFlowers:

    """Test suite for NOJ_flowers specific implementation."""

    @pytest.fixture
    def setup(self):
        """Setup NOJ_flowers model with Hornsrev1Site (12 directions, 30° bins)."""
        site = Hornsrev1Site()
        wt = V80()
        return NOJ_flowers(site=site, windTurbines=wt, n_terms=6), site, wt


    def test_noj_model_creation(self, setup):
        """
        Test NOJ model instantiation.
        Checking that the model is created successfully, has the aep_i method, and that n_terms is set correctly.
        """
        model, site, wt = setup
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 6


    def test_noj_initialization_custom_params(self):
        """
        Test NOJ model initialization with custom parameters.
        This test checks that the model can be initialized with custom values for k, n_terms, ws_cutout, and rho,
        and that these values are set correctly on the model instance.
        """
        site = Hornsrev1Site()
        wt = V80()
        model = NOJ_flowers(
            site=site, windTurbines=wt,
            k=0.05, n_terms=6, ws_cutout=24, rho=1.23
        )

        assert model.k == 0.05
        assert model.n_terms == 6
        assert model.ws_cutout == 24
        assert model.rho == 1.23


    def test_noj_aep_computation(self, setup):
        """
        Test AEP computation for NOJ model.
        Verify that the AEP is computed successfully, is a positive value, and is of the correct type (float).
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        aep = model.aep(x, y)

        assert isinstance(aep, float)
        assert aep > 0


    def test_noj_aep_i(self, setup):
        """
        Test NOJ aep_i computation.
        Check that the aep_i method returns an array of the same length as the number of turbines, and that
        all AEP values are non-negative.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        aep_i = model.aep_i(x, y)
        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)


    def test_noj_aep_i_different_positions(self, setup):
        """
        Test NOJ aep_i with various turbine positions.
        Verify that the aep_i method can handle different configurations of turbine positions and still
        returns valid AEP values.
        """
        model, site, wt = setup

        x = np.array([0, 100, 200])
        y = np.array([0, 100, 200])
        aep_i1 = model.aep_i(x, y)

        x = np.array([50, 150])
        y = np.array([50, 150])
        aep_i2 = model.aep_i(x, y)

        assert len(aep_i1) == 3
        assert len(aep_i2) == 2
        assert all(aep_i1 >= 0) and all(aep_i2 >= 0)


    def test_noj_calculate_delta_p(self, setup):
        """
        Test NOJ calculate_delta_p method.
        Check that the calculate_delta_p method returns an array of the same length as the number of turbines,
        and that all wake deficits are non-negative.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        delta_p = model._calculate_delta_p(x, y)

        assert len(delta_p) == len(x)
        assert all(delta_p >= 0)


    def test_noj_calculate_p_hat(self, setup):
        """
        Test NOJ p_hat attribute.
        Verify that p_hat is a positive float representing the free-stream power proxy.
        """
        model, site, wt = setup

        p_hat = model.p_hat

        assert isinstance(p_hat, float)
        assert p_hat > 0


    def test_noj_aep_gradient_autograd_x(self, setup):
        """
        Test NOJ AEP gradient with respect to x using autograd.
        Check that the gradient is not None and has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_x = model.aep_gradient(gradient_method="Autograd", wrt_arg="x", x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_noj_aep_gradient_autograd_y(self, setup):
        """
        Test NOJ AEP gradient with respect to y using autograd.
        Check that the gradient is not None and has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_y = model.aep_gradient(gradient_method="Autograd", wrt_arg="y", x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_noj_aep_gradient_autograd_both(self, setup):
        """
        Test NOJ AEP gradient with respect to both x and y using autograd.
        Check that the gradients are returned as a tuple and each has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_xy = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_noj_aep_gradient_exact_x(self, setup):
        """
        Test NOJ exact AEP gradient with respect to x.
        Check that the closed-form gradient is not None and has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_x = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_noj_aep_gradient_exact_y(self, setup):
        """
        Test NOJ exact AEP gradient with respect to y.
        Check that the closed-form gradient is not None and has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_y = model.aep_gradient(gradient_method="Exact", wrt_arg=["y"], x=x, y=y)

        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_noj_aep_gradient_exact_both(self, setup):
        """
        Test NOJ exact AEP gradient with respect to both x and y.
        Check that the gradients are returned as a tuple and each has the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_xy = model.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)

        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_noj_gradient_function_autograd(self, setup):
        """
        Test that NOJ returns a callable gradient function when x and y are not provided.
        Check that the returned function produces a gradient of the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_func = model.aep_gradient(gradient_method="Autograd", wrt_arg=["x"])
        grad_x = grad_func(x, y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_noj_gradient_function_exact(self, setup):
        """
        Test that NOJ returns a callable exact gradient function when x and y are not provided.
        Check that the returned function produces a gradient of the correct shape.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        grad_func = model.aep_gradient(gradient_method="Exact", wrt_arg=["x"])
        grad_x = grad_func(x, y)

        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_noj_plot_aep_per_turbine(self, setup):
        """
        Test NOJ plot_AEP_per_turbine method.
        Check that the method executes without errors when given valid x and y inputs.
        """
        model, site, wt = setup
        x, y = site.initial_position.T

        try:
            model.plot_AEP_per_turbine(x, y)
            plt.close('all')
        except Exception as e:
            pytest.fail(f"plot_AEP_per_turbine raised {type(e).__name__}: {e}")


    def test_noj_input_as_list(self, setup):
        """
        Test that NOJ accepts Python lists as x and y inputs (not just numpy arrays).
        """
        model, site, wt = setup

        aep = model.aep([0, 100, 200], [0, 100, 200])

        assert aep > 0


class TestNOJWindRoseBinSizes:

    """Test that NOJ_flowers works correctly across different wind rose bin sizes."""

    def test_noj_12dir_wind_rose(self):
        """
        Test NOJ with a 12-direction wind rose (30° bins) from Hornsrev1Site.
        Checks that AEP is positive and per-turbine array has the right length.
        """
        site = Hornsrev1Site()
        wt = V80()
        model = NOJ_flowers(site=site, windTurbines=wt, n_terms=6)
        x = np.array([0., 500., 1000.])
        y = np.array([0., 0., 0.])

        aep = model.aep(x, y)
        aep_i = model.aep_i(x, y)

        assert aep > 0
        assert len(aep_i) == len(x)


    def test_noj_360dir_wind_rose(self, generic_site):
        """
        Test NOJ with a 360-direction wind rose (1° bins).
        Checks that the model handles fine angular resolution and returns a positive AEP.
        """
        wt = V80()
        model = NOJ_flowers(site=generic_site, windTurbines=wt, n_terms=10)
        x = np.array([0., 500., 1000.])
        y = np.array([0., 0., 0.])

        aep = model.aep(x, y)

        assert aep > 0


    def test_noj_bin_size_gives_different_aep(self, generic_site):
        """
        Test that 12-direction and 360-direction wind roses give different AEP values.
        Both results must be positive; a difference confirms the model responds to wind rose resolution.
        """
        wt = V80()
        x = np.array([0., 500., 1000.])
        y = np.array([0., 0., 0.])

        model_12 = NOJ_flowers(site=Hornsrev1Site(), windTurbines=wt, n_terms=6)
        model_360 = NOJ_flowers(site=generic_site, windTurbines=wt, n_terms=10)

        aep_12 = model_12.aep(x, y)
        aep_360 = model_360.aep(x, y)

        assert aep_12 > 0
        assert aep_360 > 0
        assert aep_12 != aep_360


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
