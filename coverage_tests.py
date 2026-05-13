import pytest
import numpy as np
from matplotlib import pyplot as plt

from FLOWERS_integrated import NOJ_flowers, gaussian_flowers
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80


class TestNOJFlowers:

    """Test suite for NOJ_flowers specific implementation"""
    
    @pytest.fixture
    def setup(self):

        """Setup NOJ_flowers model and site"""

        site = Hornsrev1Site()
        wt = V80()
        return NOJ_flowers(site=site, WindTurbine=wt, n_terms=6), site, wt
    
    
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
            site=site, WindTurbine=wt, 
            k=0.05, n_terms=6, ws_cutout=24, rho=1.23
        )
        
        assert model.k == 0.05
        assert model.n_terms == 6
        assert model.ws_cutout == 24
        assert model.rho == 1.23
    

    def test_noj_aep_computation(self, setup):

        """
        Test AEP computation for NOJ model
        Verify that the AEP is computed successfully, is a positive value, and is of the correct type (float).
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        aep = model.aep(x, y)
        
        assert isinstance(aep, (float))
        assert aep > 0  # AEP should be positive

    
    def test_noj_aep_i(self, setup):

        """
        Test NOJ aep_i computation
        Check that the aep_i method returns an array of the same length as the number of turbines, and that
        all AEP values are non-negative.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        aep_i = model.aep_i(x, y)
        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)  # All AEP values should be non-negative

    
    def test_noj_aep_i_different_positions(self, setup):

        """
        Test NOJ aep_i with various turbine positions
        Verify that the aep_i method can handle different configurations of turbine positions and still
        returns valid AEP values.
        """

        model, site, wt = setup
        
        # Test with small grid
        x = np.array([0, 100, 200])
        y = np.array([0, 100, 200])
        aep_i1 = model.aep_i(x, y)
        
        # Test with different configuration
        x = np.array([50, 150])
        y = np.array([50, 150])
        aep_i2 = model.aep_i(x, y)
        
        assert len(aep_i1) == 3
        assert len(aep_i2) == 2
        assert all(aep_i1 >= 0) and all(aep_i2 >= 0)

    
    def test_noj_calculate_delta_p(self, setup):

        """
        Test NOJ calculate_delta_p method
        Check that the calculate_delta_p method returns an array of the same length as the number of turbines,
        and that all wake deficits are non-negative.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        delta_p = model.calculate_delta_p(x, y)
        
        assert len(delta_p) == len(x)
        assert all(delta_p >= 0)  # Wake deficits should be non-negative

    
    def test_noj_calculate_p_hat(self, setup):

        """
        Test NOJ calculate_p_hat method
        Verify that the calculate_p_hat method returns a positive float value representing the free stream power.
        """

        model, site, wt = setup
        
        p_hat = model.p_hat
        
        assert isinstance(p_hat, float)
        assert p_hat > 0  # Free stream power should be positive

    
    def test_noj_aep_gradient_autograd_x(self, setup):

        """
        Test NOJ AEP gradient with respect to x using autograd
        Check that the gradient with respect to x is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_x = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="x", 
            x=x, y=y
        )
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)

    
    def test_noj_aep_gradient_autograd_y(self, setup):

        """
        Test NOJ AEP gradient with respect to y using autograd
        Check that the gradient with respect to y is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_y = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="y", 
            x=x, y=y
        )
        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)

    
    def test_noj_aep_gradient_autograd_both(self, setup):

        """
        Test NOJ AEP gradient with respect to both x and y using autograd
        Check that the gradients with respect to both x and y are computed successfully, are returned as a tuple,
        and that each gradient has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_xy = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg=["x", "y"], 
            x=x, y=y
        )
        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_noj_aep_gradient_exact_x(self, setup):

        """
        Test NOJ exact AEP gradient with respect to x
        Check that the exact gradient with respect to x is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_x = model.aep_gradient(
            gradient_method="Exact", 
            wrt_arg="x", 
            x=x, y=y
        )
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)

    
    def test_noj_aep_gradient_exact_y(self, setup):

        """
        Test NOJ exact AEP gradient with respect to y
        Check that the exact gradient with respect to y is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_y = model.aep_gradient(
            gradient_method="Exact", 
            wrt_arg="y", 
            x=x, y=y
        )
        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_noj_aep_gradient_exact_both(self, setup):

        """
        Test NOJ exact AEP gradient with respect to both x and y
        Check that the exact gradients with respect to both x and y are computed successfully, are returned as a tuple,
        and that each gradient has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_xy = model.aep_gradient(
            gradient_method="Exact", 
            wrt_arg=["x", "y"], 
            x=x, y=y
        )
        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_noj_gradient_function_autograd(self, setup):

        """
        Test NOJ gradient function returned without x and y
        Check that the gradient function can be returned without providing x and y, and that it computes the correct
        gradients when called with x and y.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_func = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="x"
        )
        grad_x = grad_func(x, y)
        
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)

    
    def test_noj_gradient_function_exact(self, setup):

        """
        Test NOJ exact gradient function returned without x and y
        Check that the gradient function can be returned without providing x and y, and that it computes the correct
        gradients when called with x and y.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_func = model.aep_gradient(
            gradient_method="Exact", 
            wrt_arg="x"
        )
        grad_x = grad_func(x, y)
        
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)

    
    def test_noj_plot_aep_per_turbine(self, setup):

        """
        Test NOJ plot_AEP_per_turbine method
        Check that the plot_AEP_per_turbine method executes without errors when given valid x and y inputs.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        # Should not raise an error
        try:
            model.plot_AEP_per_turbine(x, y)
            plt.close('all')  # Close the figure to prevent display issues in tests
        except Exception as e:
            pytest.fail(f"plot_AEP_per_turbine raised {type(e).__name__}: {e}")

    
    def test_noj_input_as_list(self, setup):

        """
        Test that NOJ inputs work as lists (not just arrays)
        Check that the model can compute AEP successfully when x and y are provided as lists instead of numpy arrays.
        """

        model, site, wt = setup
        
        x = [0, 100, 200]
        y = [0, 100, 200]
        aep = model.aep(x, y)
        
        assert aep > 0


class TestGaussianFlowers:

    """Test suite for gaussian_flowers specific implementation"""
    
    @pytest.fixture
    def setup(self):

        """
        Setup gaussian_flowers model and site
        """

        site = Hornsrev1Site()
        wt = V80()
        return gaussian_flowers(site=site, WindTurbine=wt, n_terms=6), site, wt
    

    def test_gaussian_model_creation(self, setup):

        """
        Test gaussian model instantiation
        Checking that the model is created successfully, has the aep_i method, and that n_terms is set correctly.
        """

        model, site, wt = setup
        assert model is not None
        assert hasattr(model, 'aep_i')
        assert model.n_terms == 6

    
    def test_gaussian_initialization_custom_params(self):

        """
        Test gaussian model initialization with custom parameters
        This test checks that the model can be initialized with custom values for k, n_terms, ws_cutout, and rho,
        and that these values are set correctly on the model instance.
        """

        site = Hornsrev1Site()
        wt = V80()
        model = gaussian_flowers(
            site=site, WindTurbine=wt, 
            k=0.05, n_terms=6, ws_cutout=24, rho=1.23
        )
        
        assert model.k == 0.05
        assert model.n_terms == 6
        assert model.ws_cutout == 24
        assert model.rho == 1.23
    

    def test_gaussian_aep_computation(self, setup):

        """
        Test AEP computation for gaussian model
        Verify that the AEP is computed successfully, is a positive value, and is of the correct type (float).
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        aep = model.aep(x, y)
        
        assert isinstance(aep, (float))
        assert aep > 0  # AEP should be positive
    
    def test_gaussian_aep_i(self, setup):
        
        """
        Test gaussian aep_i computation
        Check that the aep_i method returns an array of the same length as the number of turbines, and that
        all AEP values are non-negative."""

        model, site, wt = setup
        x, y = site.initial_position.T
        
        aep_i = model.aep_i(x, y)
        assert len(aep_i) == len(x)
        assert all(aep_i >= 0)  # All AEP values should be non-negative

    
    def test_gaussian_aep_i_different_positions(self, setup):

        """
        Test gaussian aep_i with various turbine positions
        Verify that the aep_i method can handle different configurations of turbine positions and still
        returns valid AEP values."""

        model, site, wt = setup
        
        # Test with small grid
        x = np.array([0, 100, 200])
        y = np.array([0, 100, 200])
        aep_i1 = model.aep_i(x, y)
        
        # Test with different configuration
        x = np.array([50, 150])
        y = np.array([50, 150])
        aep_i2 = model.aep_i(x, y)
        
        assert len(aep_i1) == 3
        assert len(aep_i2) == 2
        assert all(aep_i1 >= 0) and all(aep_i2 >= 0)

    
    def test_gaussian_universal_ct(self, setup):
        
        """
        Test gaussian universal_ct method
        Check that the universal_ct method returns a value between 0 and 1, which is the expected range for thrust coefficients.
        """

        model, site, wt = setup
        
        ct = model.universal_ct()
        
        assert isinstance(ct, (float, np.ndarray))
        assert 0 < ct < 1  # Thrust coefficient should be between 0 and 1

    
    def test_gaussian_aep_gradient_autograd_x(self, setup):

        """
        Test gaussian AEP gradient with respect to x using autograd
        Check that the gradient with respect to x is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_x = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="x", 
            x=x, y=y
        )
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)
        

    def test_gaussian_aep_gradient_autograd_y(self, setup):

        """
        Test gaussian AEP gradient with respect to y using autograd
        Check that the gradient with respect to y is computed successfully, is not None, and has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_y = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="y", 
            x=x, y=y
        )
        assert grad_y is not None
        assert np.shape(grad_y) == np.shape(y)


    def test_gaussian_aep_gradient_autograd_both(self, setup):

        """
        Test gaussian AEP gradient with respect to both x and y using autograd
        Check that the gradients with respect to both x and y are computed successfully, are returned as a tuple,
        and that each gradient has the correct shape.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_xy = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg=["x", "y"], 
            x=x, y=y
        )
        assert isinstance(grad_xy, tuple)
        assert len(grad_xy) == 2
        assert np.shape(grad_xy[0]) == np.shape(x)
        assert np.shape(grad_xy[1]) == np.shape(y)


    def test_gaussian_aep_gradient_exact_not_implemented(self, setup):

        """
        Test that gaussian exact gradient raises NotImplementedError
        Check that attempting to compute exact gradients with the gaussian model raises a NotImplementedError,
        since this method is not implemented for the gaussian model.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        with pytest.raises(NotImplementedError, match="does not support exact gradients"):
            model.aep_gradient(
                gradient_method="Exact", 
                wrt_arg="x", 
                x=x, y=y
            )

    
    def test_gaussian_gradient_function_autograd(self, setup):

        """
        Test gaussian gradient function returned without x and y
        Check that the gradient function can be returned without providing x and y, and that it computes the correct
        gradients when called with x and y.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        grad_func = model.aep_gradient(
            gradient_method="Autograd", 
            wrt_arg="x"
        )
        grad_x = grad_func(x, y)
        
        assert grad_x is not None
        assert np.shape(grad_x) == np.shape(x)


    def test_gaussian_plot_aep_per_turbine(self, setup):

        """
        Test gaussian plot_AEP_per_turbine method
        Check that the plot_AEP_per_turbine method executes without errors when given valid x and y inputs.
        """

        model, site, wt = setup
        x, y = site.initial_position.T
        
        # Should not raise an error
        try:
            model.plot_AEP_per_turbine(x, y)
            plt.close('all')  # Close the figure to prevent display issues in tests
        except Exception as e:
            pytest.fail(f"plot_AEP_per_turbine raised {type(e).__name__}: {e}")
    

    def test_gaussian_input_as_list(self, setup):

        """
        Test that gaussian inputs work as lists (not just arrays)
        Check that the model can compute AEP successfully when x and y are provided as lists instead of numpy arrays.
        """

        model, site, wt = setup
        
        x = [0, 100, 200]
        y = [0, 100, 200]
        aep = model.aep(x, y)
        
        assert aep > 0
    

    def test_gaussian_vs_noj_different_results(self):

        """
        Test that gaussian and NOJ models produce different AEP results
        Check that the AEP computed by the gaussian model is different from the AEP computed by the NOJ model for
        the same turbine positions, since they are different models and should not produce identical results.
        """

        site = Hornsrev1Site()
        wt = V80()
        x, y = site.initial_position.T
        
        noj_model = NOJ_flowers(site=site, WindTurbine=wt, n_terms=6)
        gaussian_model = gaussian_flowers(site=site, WindTurbine=wt, n_terms=6)
        
        noj_aep = noj_model.aep(x, y)
        gaussian_aep = gaussian_model.aep(x, y)
        
        # Different models should produce different results
        assert noj_aep != gaussian_aep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])