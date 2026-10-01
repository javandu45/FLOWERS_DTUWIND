from abc import ABC, abstractmethod

import autograd.numpy as anp
from autograd import grad
from scipy.special import gamma
from matplotlib import pyplot as plt
import numpy as np

class FLOWERS_model(ABC):

    """
    Base class for FLOWERS models.

    FLOWERS is an AEP model to efficiently compute a wind farm's AEP. It uses numerous simplifications
    and assumptions, but most importantly a Fourier transform turning discrete components, functions of wind
    direction, into a continuous function. From this, it  solves the integral analytically, rather than 
    numerically, also enabling the computation of analytical gradients.
    """

    def __init__(self, site, windTurbines, k=0.04, n_terms=10, ws_cutout=25, rho=1.225):

        """
        Model initialization. Given its approach, FLOWERS presents the following modelling limitations:
            - All wind turbines throughout the wind farm must be of the same type.
            - Wind conditions must remain uniform in the wind farm area, atmospheric homogeneity.

        Parameters
        ----------
        Site : Site
            Site Object (UniformWeibullSite)
        windTurbine : windTurbines
            windTurbines object representing the wake generating wind turbines
        k : float
            Wake expansion coefficient, for those wake models with one, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 5-15 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3
        """

        self.site = site
        self.windTurbines = windTurbines
        self.k = k
        self.n_terms = n_terms
        self.ws_cutout = ws_cutout
        self.rho = rho

        # Frquency distribution and average wind speed for each wind direction for the location
        self.avg_ws = site.ds.Weibull_A.values * gamma(1 + 1/site.ds.Weibull_k.values)
        self.freqs = site.ds.Sector_frequency.values
        self.freqs = self.freqs / sum(self.freqs)

        # Average wind speed normalization with cut out wind speed
        self.avg_ws_norm = anp.array(self.avg_ws) / self.ws_cutout

        # Power and thrust coefficients for each wind direction (average wind speed per bin)
        # I could not find any other way of obtaining the power coefficient
        ideal_power = 1/2 * self.rho * self.windTurbines.diameter()**2/4 * anp.pi * self.avg_ws**3
        self.cp = self.windTurbines.power(self.avg_ws)/ideal_power
        self.ct = self.windTurbines.ct(self.avg_ws)  


    def aep(self, x, y):

        """
        Computes the wind farm's AEP using FLOWERS model, defined as the summation of the contribution 
        from each turbine

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

        Returns
        -------
        aep : float
            Wind farm's AEP (in GWh) using FLOWERS model
        """

        # AEP per turbine
        aep_i = self.aep_i(x, y)

        # AEP of the entire wind farm
        aep = anp.sum(aep_i)

        return aep
    

    def _fourier_coefficients(self, values):

        """
        Obtain Fourier coefficients to transform discrete components, which are a function of wind direction,
        into continuous form.

        Parameters
        ----------
        values : array_like
            Array of length n_wd containing the values of the function to be transformed for each wind direction

        Returns
        -------
        fc : dict
            a : Array of length n_terms containing Fourier coefficient a
            b : Array of length n_terms containing Fourier coefficient b
            m : Array of length n_terms containing the coefficient index
        """

        n_terms = self.n_terms

        # Fourier coefficients: a_0, a_m, b_m
        coeffs = 2 * anp.fft.rfft(values)
        a = coeffs.real
        b = -coeffs.imag

        a_sorted = np.argsort(np.abs(a))
        b_sorted = np.argsort(np.abs(b))
        # m = anp.arange(len(a))

        # Use only the first n_terms
        if values.ndim == 1:
            if n_terms > 0 and n_terms <= len(a):
                # a = a[b_sorted[-n_terms:]]
                a = a[0:n_terms]
                # b = b[b_sorted[-n_terms:]]
                b = b[0:n_terms]
                m = anp.arange(n_terms)
                # m = m[b_sorted[-n_terms:]]
                fc = {"a": a, "b": b, "m": m}

                return fc

            else:
                raise ValueError("n_terms should be between 0 and n_wd/2")
            
        elif values.ndim == 2:
            if n_terms > 0 and n_terms <= values.shape[1]:
                a = a[..., 0:n_terms]
                b = b[:, 0:n_terms]
                m = anp.arange(n_terms)
                fc = {"a": a, "b": b, "m": m}

                return fc

            else:
                raise ValueError("n_terms should be between 0 and n_wd/2")

    
    @abstractmethod
    def aep_i(self, x, y):
        """
        Computes AEP for each turbine (i)

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

        Returns
        -------
        aep_i : array_like
            AEP contribution from each turbine (in GWh)
        """
    
    
    def _aep_gradient_autograd(self, wrt_arg=['x', 'y'], x=None, y=None):

        """
        Compute the AEP gradients with respect to the turbine positions x and y using automatic differentiation.
        
        Parameters
        ----------
        wrt_arg : list or str, optional
            Arguments with respect to which to compute gradients. Can be 'x', 'y', ['x'], ['y'],
            or ['x', 'y']. Default is ['x', 'y']
        x : array_like, optional
            x-coordinates of the turbines
        y : array_like, optional
            y-coordinates of the turbines

        Returns
        -------
        If x and y are provided:
            Gradients of AEP with respect to requested arguments. Returns single array if one
            argument requested, tuple if multiple arguments requested.
        If x and y are not provided:
            gradient_function : callable
            Function that computes gradients when called with x and y
        """

        wrt_arg = tuple(np.atleast_1d(wrt_arg))

        grad_fun = grad(
            self.aep,
            argnum=tuple({'x': 0, 'y': 1}[arg] for arg in wrt_arg)
        )

        def evaluate(x, y):
            result = grad_fun(
                anp.asarray(x, float),
                anp.asarray(y, float)
            )
            return result[0] if len(wrt_arg) == 1 else tuple(result)

        return evaluate if x is None and y is None else evaluate(x, y)
    
    
    def _aep_gradient_exact(self, wrt_arg, x, y):

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exact gradients. "
            f"Use method='Autograd' instead."
        )


    def aep_gradient(self, gradient_method="Autograd", wrt_arg=['x', 'y'], x=None, y=None):

        # I thought about importing pywake's autograd, but then how would it use the exact gradients

        """
        Compute the AEP gradients with respect to the turbine positions x and y, using either 
        automatic differentiation or analytical gradients (if implemented for the specific model).
        
        This method has two behaviours:
        1) Without specifying x and y, returns the function to compute the gradients:
        gradient_function = wfm.aep_gradient(wrt_arg=['y'])
        dy = gradient_function(x, y)
        
        2) With x and y specified, computes and returns the gradients:
        dy = wfm.aep_gradient(wrt_arg=['y'], x=x, y=y)

        Parameters
        ----------
        gradient_method : {"Autograd", "Exact"}, optional
            The method to use for computation. Default is "Autograd".
            - "Autograd": Uses automatic differentiation using the autograd package.
            - "Exact": Uses exact analytical gradients.
        wrt_arg : list or str, optional
            Arguments with respect to which to compute gradients. Can be ['x'], ['y'],
            or ['x', 'y']. Default is ['x', 'y']
        x : array_like, optional
            x-coordinates of the turbines
        y : array_like, optional
            y-coordinates of the turbines

        Returns
        -------
        If x and y are provided:
            Gradients of AEP with respect to requested arguments. Returns single array if one
            argument requested, tuple if multiple arguments requested.
        If x and y are not provided:
            gradient_function : callable
            Function that computes gradients when called with x and y
        """            

        if gradient_method == "Autograd":
            return self._aep_gradient_autograd(wrt_arg=wrt_arg, x=x, y=y)
        
        if gradient_method == "Exact":
            if x is not None and y is not None:
                return self._aep_gradient_exact(wrt_arg=wrt_arg, x=x, y=y)
            else:
                def gradient_function(x, y):
                    return self._aep_gradient_exact(wrt_arg=wrt_arg, x=x, y=y)
                return gradient_function

    
    def plot_AEP_per_turbine(self, x, y):

        """
        Plots the AEP contribution from each turbine in the wind farm

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines     
        """

        aep_turbines = self.aep_i(x, y)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c=aep_turbines, cmap='viridis', s=100)
        plt.colorbar(label='AEP per Turbine (GWh)')
        # Set title depending on specific FLOWERS subclass
        cls_name = self.__class__.__name__
        if cls_name == 'NOJ_flowers':
            title = 'AEP per turbine using FLOWERS Model (NOJ)'
        elif cls_name == 'gaussian_flowers':
            title = 'AEP per turbine using FLOWERS Model (Gaussian)'
        elif cls_name == 'TurbOPark_flowers':
            title = 'AEP per turbine using FLOWERS Model (TurbOPark)'
        elif cls_name == 'fuga_flowers':
            title = 'AEP per turbine using FLOWERS Model (Fuga)'
        else:
            title = 'AEP per Turbine using Flowers Model'
        plt.title(title)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.show()
