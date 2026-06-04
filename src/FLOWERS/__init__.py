from abc import ABC, abstractmethod

import autograd.numpy as anp
from autograd import grad
from scipy.special import gamma
from matplotlib import pyplot as plt

import numpy as np

class FLOWERS_model(ABC):

    """
    Base class for FLOWERS models.

    FLOWERS is an AEP model which efficiently computes a wind farm's AEP. It uses numerous simplifications
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
            Wake expansion coefficient, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
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

        # Use only the first n_terms
        if n_terms > 0 and n_terms <= len(a):
            a = a[0:n_terms]
            b = b[0:n_terms]
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
        else:
            title = 'AEP per Turbine using Flowers Model'
        plt.title(title)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.show()


class NOJ_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - NO Jensen wake model.

    Based on "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization"
    https://doi.org/10.1002/we.2954
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
            Wake expansion coefficient, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3
        """

        super().__init__(site, windTurbines, k, n_terms, ws_cutout, rho)

        # Fourier coefficients
        fourier_function = self.cp**(1/3) * self.avg_ws_norm * (1-anp.sqrt(1-self.ct)) * self.freqs
        self.fc = self._fourier_coefficients(fourier_function)

        # Free stream AEP component for a single turbine (p_hat) - Equation 18
        self.p_hat = self._calculate_p_hat()


    def _calculate_p_hat(self):

        """
        Free stream AEP component for a single turbine - Equation 18

        Returns
        -------
        p_hat : float

        """

        # Free stream AEP component for a single turbine - Equation 18
        p_hat = anp.sum(self.cp**(1/3) * self.avg_ws_norm * self.freqs)

        return float(p_hat)
    

    def _calculate_delta_p(self, x, y):

        """
        Computes dimensionless wake losses coming from all turbines j on each turbine i for the layout given by
        x and y coordinates.

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

            
        Returns
        -------
        delta_p : float
            An array of size len(x) containing the addition of the wake loses coming from all turbines j 
            on each turbine i 

        """

        D = self.windTurbines.diameter()

        epsilon = 1e-12  # To avoid division by zero

        x = anp.array(x)
        y = anp.array(y)

        # Normalized relative position between turbines i and j
        xij = (x[None, :] - x[:, None])/D
        yij = (y[None, :] - y[:, None])/D

        # Transform to polar coordinates
        r_ij_hat = anp.sqrt(xij**2 + yij**2)
        theta_ij = anp.nan_to_num(anp.arctan2(yij, xij))

        r_ij_hat = anp.where(r_ij_hat == 0, 1e-6, r_ij_hat)  # To avoid numerical issues

        # Normalize polar coordinates with respect to rotor diameter
        theta_ij_hat = theta_ij / (2*anp.pi)

        # Critical polar angle of wake edge (theta_c) - Equation 11
        theta_c = anp.nan_to_num(anp.arctan((1/(2*r_ij_hat + epsilon) + self.k * anp.sqrt(1 + self.k**2 - (1/(2*r_ij_hat)**2))) / 
                          (-self.k/(2*r_ij_hat + epsilon) + anp.sqrt(1 + self.k**2 - (1/(2*r_ij_hat + epsilon)**2)))) / (2 * anp.pi))

        # ---------------------------------------------------------------------------
        # Wake loss component - Eequation 28
        # Zero order component
        delta_p = self.fc["a"][0] * theta_c / (2 * self.k * r_ij_hat + 1)**2 * (
                    1 + 8 * anp.pi**2 * self.k * r_ij_hat * theta_c**2 / (
                    3*(2 * self.k * r_ij_hat + 1)))
        
        # Preparing variables for vectorized computation
        theta_ij_hat = theta_ij_hat[:,:,None]
        r_ij_hat = r_ij_hat[:,:,None]
        theta_c = theta_c[:,:,None]
        m = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, 1:]
        a = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, 1:]
        b = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, 1:]

        # Higher order components
        # Precompute trigonometric terms
        two_pi_m_theta_ij = 2 * anp.pi * m * theta_ij_hat
        two_pi_m_theta_c = 2 * anp.pi * m * theta_c
        
        cos_theta_ij = anp.cos(two_pi_m_theta_ij)
        sin_theta_ij = anp.sin(two_pi_m_theta_ij)
        cos_theta_c = anp.cos(two_pi_m_theta_c)
        sin_theta_c = anp.sin(two_pi_m_theta_c)
        
        # Precompute common terms
        two_k_r_plus_one = 2 * self.k * r_ij_hat + 1
        m_squared = m**2
        
        outside_brackets = (a * cos_theta_ij + b * sin_theta_ij) / (
                    anp.pi * m * two_k_r_plus_one**2)
        
        inside_brackets = sin_theta_c + 2 * self.k * r_ij_hat * (
                    (two_pi_m_theta_c**2 - 2) * sin_theta_c +
                    4*anp.pi * m * theta_c * cos_theta_c) / (
                    m_squared * two_k_r_plus_one)

        # Summing over all m-Fourier terms
        delta_p = anp.nan_to_num(delta_p) + anp.sum(anp.nan_to_num(outside_brackets * inside_brackets), axis=2)
        # ---------------------------------------------------------------------------

        # Sum wake contribution over all turbines (j)
        delta_p = anp.sum(delta_p, axis=1)

        return delta_p


    def aep_i(self, x, y):

        """
        Computes the AEP contribution from each turbine (i), which is the result of substracting all wake interactions
        experienced by turbine i (delta_p) from the free stream AEP component for a turbine (p_hat)
        
        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

        Returns
        -------
        aep_i : array_like
            AEP contribution from each turbine
        """

        # Free stream AEP component for a single wind turbine - Equation 18
        p_hat = self.p_hat

        # Wake loss component - Equation 28
        delta_p = self._calculate_delta_p(x, y)

        # AEP contribution from each turbine i (freestream AEP - wakes from all turbines on turbine i)
        aep_turbine = (p_hat - delta_p)**3

        # Return dimensions to AEP
        aep_turbine = aep_turbine * 8760 * anp.pi/8 * self.rho * self.windTurbines.diameter()**2 * (self.ws_cutout**3)/1e9

        return aep_turbine


    def _aep_gradient_exact(self, x, y,  wrt_arg= ['x', 'y']):

        """
        Compute the AEP gradients with respect to the turbine positions x and y.

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines
        wrt_arg : list or str, optional
            Arguments with respect to which to compute gradients. Can be 'x', 'y', ['x'], ['y'], or ['x', 'y']. Default is ['x', 'y']

        Returns
        -------
        daep_dx : np.array
            Array of length n containing the AEP gradients with respect to the x-coordinates
        daep_dy : np.array
            Array of length n containing the AEP gradients with respect to the y-coordinates
        
        """

        RotorDiameter = self.windTurbines.diameter()

        # Relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None])/RotorDiameter
        yij = (y[None, :] - y[:, None])/RotorDiameter

        # Transform to polar coordinates
        r_ij_hat = anp.sqrt(xij**2 + yij**2)
        theta_ij_hat = anp.arctan2(yij, xij) / (2 * anp.pi)

        # Critical polar angle of wake edge (theta_c) - Equation 11
        inside_sqrt = 1 + self.k**2 - (1/(2*r_ij_hat))**2        
        top = 1/(2*r_ij_hat) + self.k * anp.sqrt(inside_sqrt)
        bottom = -self.k/(2*r_ij_hat) + anp.sqrt(inside_sqrt)
        theta_c = anp.nan_to_num(anp.arctan(top/bottom)/(2 * anp.pi))

        # Derivate of theta_c (critical angle) with respect to r_ij_hat (normalized polar coordinate) - Equation 41
        inside_sqrt = self.k**2 - (1 / (2*r_ij_hat))**2 + 1
        dtheta_c_dr = -1 / (4 * anp.pi * r_ij_hat**2 * anp.sqrt(inside_sqrt))

        # ---------------------------------------------------------------------------
        # DERIVATIVE OF WAKE LOSS COMPONENT (delta_p) WITH RESPECT TO THE RADIUS (r_ij_hat)
        # Zero frequency component of the gradient with respect to r_ij_hat - Eequation 42
        inside_brackets = -4 * self.k * theta_c * (3 + 6 * self.k * r_ij_hat + 2 * anp.pi**2 * theta_c**2 * (4 * self.k * r_ij_hat - 1)) + \
                            3 * (2 * self.k * r_ij_hat + 1) * (1 + 2 * self.k * r_ij_hat + 8 * anp.pi**2 * self.k * r_ij_hat * theta_c**2) * dtheta_c_dr
        outside_brackets = self.fc["a"][0] / (3 * (2 * self.k * r_ij_hat + 1)**4)
        ddelta_p_dr = anp.nan_to_num(outside_brackets * inside_brackets)

        # Preparing variables for vectorized computation
        theta_ij_hat = theta_ij_hat[:,:,None]
        r_ij_hat = r_ij_hat[:,:,None]
        theta_c = theta_c[:,:,None]
        dtheta_c_dr = dtheta_c_dr[:,:,None]
        m = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, 1:]
        a = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, 1:]
        b = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, 1:]

        # Precompute trigonometric terms
        two_pi_m_theta_ij = 2 * anp.pi * m * theta_ij_hat
        two_pi_m_theta_c = 2 * anp.pi * m * theta_c
        
        cos_theta_ij = anp.cos(two_pi_m_theta_ij)
        sin_theta_ij = anp.sin(two_pi_m_theta_ij)
        cos_theta_c = anp.cos(two_pi_m_theta_c)
        sin_theta_c = anp.sin(two_pi_m_theta_c)
        
        # Precompute common terms
        two_k_r_plus_one = 2 * self.k * r_ij_hat + 1
        m_squared = m**2
        k_r = self.k * r_ij_hat
        
        # Higher order components of the gradient with respect to r_ij_hat - Equation 42
        A = 1 / (anp.pi * m**3 * two_k_r_plus_one**4)
        B = a * cos_theta_ij + b * sin_theta_ij
        C = -4 * self.k * sin_theta_c * (
                1 + m_squared + 2 * k_r * (m_squared - 2) + 2 * anp.pi**2 * m_squared * theta_c**2 * (4 * k_r - 1))
        D = 2 * anp.pi * m * cos_theta_c * (
                4 * self.k * theta_c * (1 - 4 * k_r) + \
                m_squared * two_k_r_plus_one * (1 + 2 * k_r + 8 * anp.pi**2 * k_r * theta_c**2) * dtheta_c_dr)

        # Sum over all m-Fourier terms
        ddelta_p_dr += anp.sum(anp.nan_to_num(A * B * (C + D)), axis=2)

        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # DERIVATIVE OF WAKE LOSS COMPONENT (delta_p) WITH RESPECT TO THE ANGLE (theta_ij_hat)

        # Higher order derivatives with respect to theta_ij_hat - Equation 40
        A = 2 / two_k_r_plus_one**2 * (b * cos_theta_ij - a * sin_theta_ij)
        B = sin_theta_c + \
            2 * k_r / (m_squared * two_k_r_plus_one) * (
                ((2 * anp.pi * self.k * m * theta_c) ** 2 - 2) * sin_theta_c + 4 * anp.pi * m * theta_c * cos_theta_c
            )
        
        ddelta_p_dtheta = anp.sum(anp.nan_to_num(A * B), axis=2)
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # OBTAINING THE GRADIENTS IN CARTESIAN COORDINATES
        # Obtaining free stream power and wake deficits
        p_hat = self.p_hat
        delta_p = self._calculate_delta_p(x, y)

        multiplier = (p_hat - delta_p)**2

        # Derivatives with respect to x and y coordinates - Equations 38 and 39 decomposed
        term_x = anp.nan_to_num(ddelta_p_dr * xij / (r_ij_hat[:,:,0]) - ddelta_p_dtheta * yij / (2 * anp.pi * r_ij_hat[:,:,0]**2))
        term_y = anp.nan_to_num(ddelta_p_dr * yij / (r_ij_hat[:,:,0]) + ddelta_p_dtheta * xij / (2 * anp.pi * r_ij_hat[:,:,0]**2))

        # Derivatives to account for the movement of the individual windTurbines - Equation 33
        dF_dx = anp.zeros((len(x), 1))
        dF_dy = anp.zeros((len(x), 1))

        # Applying partial derivative with respect to the movement of each individual wind windTurbines g - Equation 33
        for i in range(len(dF_dx)):
            grad_mask = anp.zeros_like(r_ij_hat[:,:,0])
            grad_mask[i,:] = -1.  # Equivalent to dxij_dxg
            grad_mask[:,i] = 1.   # Equivalent to dyij_dyg

            # Dimensionaless derivative - Equations 38 and 39
            dF_dx[i] = -3*anp.sum(multiplier *  anp.sum(term_x * grad_mask, axis=1))
            dF_dy[i] = -3*anp.sum(multiplier * anp.sum(term_y * grad_mask, axis=1))
        # ---------------------------------------------------------------------------

        # Return dimesions to gradients
        daep_dx = (dF_dx * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9
        daep_dy = (dF_dy * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9

        if wrt_arg == ['x', 'y']:
            return daep_dx.flatten(), daep_dy.flatten()
        elif wrt_arg == ['x']:
            return daep_dx.flatten()
        elif wrt_arg == ['y']:
            return daep_dy.flatten()


class gaussian_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - Gaussian Bastankhah wake model.

    Based on "Gaussian FLOWERS: Wind-rose-based analytical integration of Gaussian wake model for extremely fast
    AEP estimation"
    https://doi.org/10.1063/5.0245886
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
            Wake expansion coefficient, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3
        """

        super().__init__(site, windTurbines, k, n_terms, ws_cutout, rho)

        # Fourier coefficients
        fourier_function = len(self.freqs) / (2*anp.pi) * self.freqs * self.cp * self.avg_ws**3
        self.fc = self._fourier_coefficients(fourier_function)
        self.fc["a"] = self.fc["a"] / len(fourier_function)
        self.fc["b"] = self.fc["b"] / len(fourier_function)

        A_0 = self.fc["a"][0]/2

        A = anp.sqrt(self.fc["a"][1:] ** 2 + self.fc["b"][1:] ** 2)
        B = -anp.arctan2(self.fc["b"][1:], self.fc["a"][1:])

        self.fc["a"] = anp.concatenate([[A_0], A])
        self.fc["b"] = anp.concatenate([[0], B])

        # Getting universal thrust coefficient
        self.CT = self._universal_ct()

        # Gaussian wake parameters, based on universal thrust coefficient
        self.beta = (1 + anp.sqrt(1 - self.CT)) / (2 * anp.sqrt(1 - self.CT))
        self.epsilon = 0.2 * anp.sqrt(self.beta)

        # Limit to avoid NaNs in gaussian wake deficit
        self.lim = 1 / self.k * anp.sqrt(self.CT / 8) - self.epsilon


    def _universal_ct(self):

        """
        Computes the universal thrust coefficient used for all wind turbines for each wind direaction
        
        Returns
        -------
        CT : float
            Universal thrust coefficient

        """

        # Universal thrust coefficient - Equation 9
        CT = anp.sum(self.ct * self.freqs * self.cp * self.avg_ws**3 / (anp.sum(self.cp * self.avg_ws**3 * self.freqs)))
        
        return CT

    
    def aep_i(self, x, y):

        """
        Computes the AEP contribution from each turbine (i)

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

        Returns
        -------
        aep_per_turbine : array_like
            AEP contribution from each turbine (in GWh)

        """

        x = anp.array(x)
        y = anp.array(y)

        # Relative position between turbines i and j (n and m in the paper)
        xij = (x[:, None] - x[None, :])
        yij = (y[:, None] - y[None, :])

        # Transform to polar coordinates
        r_ij = anp.sqrt(xij**2 + yij**2)
        theta_ij = anp.arctan2(xij, yij) - anp.pi

        # Adapt theta_ij to wrap the angle into the range [-pi, pi] - m(theta) Equation 15
        theta_ij = anp.mod(theta_ij + anp.pi, 2 * anp.pi) - anp.pi

        # Gaussian wake deficit, decomposed into g and h - Equation 14
        inside_sqrt = anp.where(anp.abs(r_ij) < self.lim, 1, 1 - self.CT / (8 * (self.k * r_ij + self.epsilon)**2))
        g = 1 - anp.sqrt(inside_sqrt)
        sigma_a = (self.k * r_ij + self.epsilon) / (r_ij + 1e-10)

        # Preparing variables for vectorized computation
        theta_ij = theta_ij[:,:,None]
        g = g[:,:,None]
        sigma_a = sigma_a[:,:,None]
        t = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, :]
        A = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, :]
        PHI = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, :]

        # AEP caculation, broken into three parts - Equation 16
        constant = A[0:] * anp.cos(t * theta_ij + PHI[0:])

        inside_exp = -sigma_a**2 * t**2

        # Part 1 - Equation 18
        I0 = 2 * anp.pi * self.fc["a"][0]

        # Part 2 - Equation 19 for alpha = 1
        I1 = g * anp.sqrt(2 * anp.pi) * sigma_a * anp.exp(inside_exp / 2) * constant

        # Part 3 - Equation 19 for alpha = 2
        I2 = g ** 2 * anp.sqrt(4 * anp.pi) * sigma_a / 2 * anp.exp(inside_exp / 4) * constant

        # Sum over all fourier terms m
        aep_turbine = anp.sum(- 3 * I1 + 3 * I2, axis=-1)

        # Sum over all turbines j
        aep_turbine = anp.sum(aep_turbine, axis=-1)

        # Sum over all turbines i (adding I0 component), dimensionless AEP
        aep_turbine = aep_turbine + I0

        # Return dimensions to AEP, in GWh
        aep_turbine = aep_turbine * 0.5 * 8760 * self.rho * self.windTurbines.diameter()**2/4 * anp.pi / 1e9

        return aep_turbine


class TurbOPark_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - TurbOPark wake model.

    Similar procedure as the one in "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization"
    https://doi.org/10.1002/we.2954, all referenced equations refer to this paper.

    TurbOPark wake model is a top-hat model where the diameter of the wake does not grow linearly with distance
    based on a wake coefficient k, as in the Park model. It's growth is linked to the turbulence intensity, both 
    atmospheric and turbine-induced.

    To account for this, the implementation in FLOWERS is exactly the same as for NO Jensen, with the difference
    of k, which now is computed as k_eff, defined as the expansion coefficient that yields the same wake diameter
    as TurbOPark for a specific turbine downstream, therefore there is a k_eff for each turbine pair.
    """

    def __init__(self, site, windTurbines, n_terms=10, ws_cutout=25, rho=1.225, ti=0.1):

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
            Wake expansion coefficient, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3
        """

        k = None # k is not a parameter to be defined in TurbOPark

        super().__init__(site, windTurbines, k, n_terms, ws_cutout, rho)

        # Universal thrust coefficient
        self.CT = self._universal_ct()

        # Fourier coefficients
        fourier_function = self.cp**(1/3) * self.avg_ws_norm * (1-anp.sqrt(1-self.CT)) * self.freqs
        self.fc = self._fourier_coefficients(fourier_function)

        # Free stream AEP component for a single turbine (p_hat) - Equation 18
        self.p_hat = self._calculate_p_hat()

        # TurbOPark spceific variables
        self.I0 = ti
        self.A = 0.6
        c1 = 1.5
        c2 = 0.8
        self.alpha = c1 * self.I0
        self.beta = c2 * self.I0 / anp.sqrt(self.CT)


    def _universal_ct(self):

        """
        Computes the universal thrust coefficient used for all wind turbines for each wind direaction
        
        Returns
        -------
        CT : float
            Universal thrust coefficient

        """

        # Universal thrust coefficient - Similar to Gaussian FLOWERS
        CT = anp.sum(self.ct * self.freqs * self.cp * self.avg_ws**3 / (anp.sum(self.cp * self.avg_ws**3 * self.freqs)))
        
        return CT


    def _calculate_p_hat(self):

        """
        Free stream AEP component for a single turbine - Equation 18

        Returns
        -------
        p_hat : float

        """

        # Free stream AEP component for a single turbine - Equation 18
        p_hat = anp.sum(self.cp**(1/3) * self.avg_ws_norm * self.freqs)

        return float(p_hat)
    

    def _calculate_delta_p(self, x, y):

        """
        Computes dimensionless wake losses coming from all turbines j on each turbine i for the layout given by
        x and y coordinates.

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

            
        Returns
        -------
        delta_p : float
            An array of size len(x) containing the addition of the wake loses coming from all turbines j 
            on each turbine i 

        """

        D = self.windTurbines.diameter()

        epsilon = 1e-12  # To avoid division by zero

        x = anp.array(x)
        y = anp.array(y)

        # Normalized relative position between turbines i and j
        xij = (x[None, :] - x[:, None])/D
        yij = (y[None, :] - y[:, None])/D

        # Transform to polar coordinates
        r_ij_hat = anp.sqrt(xij**2 + yij**2)
        theta_ij = anp.nan_to_num(anp.arctan2(yij, xij))

        r_ij_hat = anp.where(r_ij_hat == 0, 1e-6, r_ij_hat)  # To avoid numerical issues

        # Obtain effective wake expansion coefficient for each turbine pair (i,j)
        k_eff_ij = self._get_parwise_k_eff(r_ij_hat)

        # Normalize polar coordinates with respect to rotor diameter
        theta_ij_hat = theta_ij / (2*anp.pi)

        # Critical polar angle of wake edge (theta_c) - Equation 11
        theta_c = anp.nan_to_num(anp.arctan((1/(2*r_ij_hat + epsilon) + k_eff_ij * anp.sqrt(1 + k_eff_ij**2 - (1/(2*r_ij_hat)**2))) / 
                          (-k_eff_ij/(2*r_ij_hat + epsilon) + anp.sqrt(1 + k_eff_ij**2 - (1/(2*r_ij_hat + epsilon)**2)))) / (2 * anp.pi))

        # ---------------------------------------------------------------------------
        # Wake loss component - Eequation 28
        # Zero order component
        delta_p = self.fc["a"][0] * theta_c / (2 * k_eff_ij * r_ij_hat + 1)**2 * (
                    1 + 8 * anp.pi**2 * k_eff_ij * r_ij_hat * theta_c**2 / (
                    3*(2 * k_eff_ij * r_ij_hat + 1)))
        
        # Preparing variables for vectorized computation
        theta_ij_hat = theta_ij_hat[:,:,None]
        r_ij_hat = r_ij_hat[:,:,None]
        theta_c = theta_c[:,:,None]
        k_eff_ij = k_eff_ij[:,:,None]
        m = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, 1:]
        a = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, 1:]
        b = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, 1:]

        # Higher order components
        # Precompute trigonometric terms
        two_pi_m_theta_ij = 2 * anp.pi * m * theta_ij_hat
        two_pi_m_theta_c = 2 * anp.pi * m * theta_c
        
        cos_theta_ij = anp.cos(two_pi_m_theta_ij)
        sin_theta_ij = anp.sin(two_pi_m_theta_ij)
        cos_theta_c = anp.cos(two_pi_m_theta_c)
        sin_theta_c = anp.sin(two_pi_m_theta_c)
        
        # Precompute common terms
        two_k_r_plus_one = 2 * k_eff_ij * r_ij_hat + 1
        m_squared = m**2
        
        outside_brackets = (a * cos_theta_ij + b * sin_theta_ij) / (
                    anp.pi * m * two_k_r_plus_one**2)
        
        inside_brackets = sin_theta_c + 2 * k_eff_ij * r_ij_hat * (
                    (two_pi_m_theta_c**2 - 2) * sin_theta_c +
                    4*anp.pi * m * theta_c * cos_theta_c) / (
                    m_squared * two_k_r_plus_one)

        # Summing over all m-Fourier terms
        delta_p = anp.nan_to_num(delta_p) + anp.sum(anp.nan_to_num(outside_brackets * inside_brackets), axis=2)
        # ---------------------------------------------------------------------------

        # Sum wake contribution over all turbines (j)
        delta_p = anp.sum(delta_p, axis=1)

        return delta_p


    def _obtain_D_w(self, r_ij_hat):

        """
        Compute the wake diameter D_w based on TurbOPark model, for a turbine assumed directly downsream of
        another turbine at a distance r_ij_hat (normalized with respect to rotor diameter), assuming 
        theta_ij_hat = 0

        Parameters
        ----------
        r_ij_hat : array_like
            Normalized distance between turbines i and j (r_ij_hat)

        Returns
        -------
        D_w : array_like
            Wake diameter D_w normalized with respect to rotor diameter
        """

        common = self.alpha + self.beta*r_ij_hat
        out_ln = anp.sqrt(common**2 + 1) - anp.sqrt(1 + self.alpha**2)
        up_ln = (anp.sqrt(common**2 + 1) + 1) * self.alpha
        down_ln = (anp.sqrt(1 + self.alpha**2) + 1) * common
        D_w = 1 + self.A*self.I0/self.beta * (out_ln - anp.log(up_ln/down_ln))

        return D_w


    def _get_parwise_k_eff(self, r_ij_hat):

        """
        Compute the effective wake expansion coefficient (k_eff) for each turbine pair (i,j) based on the wake diameter
        obtained with the TurbOPark model. k_eff is defined as k_eff = (D_w - 1) / (2 * r_ij_hat)

        Parameters
        ----------
        r_ij_hat : array_like
            Normalized distance between turbines i and j (r_ij_hat)

        Returns
        k_eff_ij : array_like
            Effective wake expansion coefficient for each turbine pair (i,j)
        """

        D_w_ij = self._obtain_D_w(r_ij_hat)
        k_eff_ij = (D_w_ij - 1) / (2 * r_ij_hat)

        return k_eff_ij


    def aep_i(self, x, y):

        """
        Computes the AEP contribution from each turbine (i), which is the result of substracting all wake interactions
        experienced by turbine i (delta_p) from the free stream AEP component for a turbine (p_hat)
        
        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines

        Returns
        -------
        aep_i : array_like
            AEP contribution from each turbine
        """

        # Free stream AEP component for a single wind turbine - Equation 18
        p_hat = self.p_hat

        # Wake loss component - Equation 28
        delta_p = self._calculate_delta_p(x, y)

        # AEP contribution from each turbine i (freestream AEP - wakes from all turbines on turbine i)
        aep_turbine = (p_hat - delta_p)**3

        # Return dimensions to AEP
        aep_turbine = aep_turbine * 8760 * anp.pi/8 * self.rho * self.windTurbines.diameter()**2 * (self.ws_cutout**3)/1e9

        return aep_turbine


    def _aep_gradient_exact(self, x, y, wrt_arg=['x', 'y']):
    
        """
        Compute the AEP gradients with respect to the turbine positions x and y.

        Parameters
        ----------
        x : array_like
            x-coordinates of the turbines
        y : array_like
            y-coordinates of the turbines
        wrt_arg : list or str, optional
            Arguments with respect to which to compute gradients. Can be 'x', 'y', ['x'], ['y'], or ['x', 'y']. Default is ['x', 'y']

        Returns
        -------
        daep_dx : np.array
            Array of length n containing the AEP gradients with respect to the x-coordinates
        daep_dy : np.array
            Array of length n containing the AEP gradients with respect to the y-coordinates
        """

        RotorDiameter = self.windTurbines.diameter()

        # Relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None]) / RotorDiameter
        yij = (y[None, :] - y[:, None]) / RotorDiameter

        # Transform to polar coordinates
        r_ij_hat = anp.sqrt(xij**2 + yij**2)
        theta_ij_hat = anp.arctan2(yij, xij) / (2 * anp.pi)

        # Obtain effective wake expansion coefficient for each turbine pair (i,j)
        k_eff_ij = self._get_parwise_k_eff(r_ij_hat)

        # Critical polar angle of wake edge (theta_c) - Equation 11
        inside_sqrt = 1 + k_eff_ij**2 - (1 / (2 * r_ij_hat))**2
        top = 1 / (2 * r_ij_hat) + k_eff_ij * anp.sqrt(inside_sqrt)
        bottom = -k_eff_ij / (2 * r_ij_hat) + anp.sqrt(inside_sqrt)
        theta_c = anp.nan_to_num(anp.arctan(top / bottom) / (2 * anp.pi))

        # Derivative of k_eff with respect to r_ij_hat, needed for the TurbOPark-specific component of the gradient.
        common = self.alpha + self.beta * r_ij_hat
        dkeff_dr = self.A * self.I0 / (2 * r_ij_hat) * (common / (np.sqrt(common**2 + 1) + 1) + 1 / common) - k_eff_ij / r_ij_hat

        # Derivative of theta_c (critical angle) with respect to r_ij_hat, holding k_eff fixed
        inside_sqrt = k_eff_ij**2 - (1 / (2 * r_ij_hat))**2 + 1
        dtheta_c_dr = - 1 / (4 * anp.pi * r_ij_hat**2 * anp.sqrt(inside_sqrt))

        # Derivative of theta_c with respect to k_eff, holding r_ij_hat fixed.
        inside_sqrt = 1 + k_eff_ij**2 - (1 / (2 * r_ij_hat))**2
        cos_alpha_c = (-k_eff_ij / (2 * r_ij_hat) + anp.sqrt(inside_sqrt)) / (1 + k_eff_ij**2)
        dtheta_c_dkeff = cos_alpha_c / (2 * anp.pi * anp.sqrt(inside_sqrt))

        # ---------------------------------------------------------------------------
        # DERIVATIVE OF WAKE LOSS COMPONENT (delta_p) WITH RESPECT TO THE RADIUS (r_ij_hat)
        # For TurbOPark, ddelta_p/dr_ij = ddelta_p/dr_ij(NOJ) + (ddelta_p/dk_eff + ddelta_p/dtheta_c * dtheta_c/dk_eff) * dk_eff/dr_ij

        # -----------------
        # ddelta_p/dr_ij_hat for the NOJ component
        # Zero frequency component of the gradient with respect to r_ij_hat - Equation 42
        inside_brackets = -4 * k_eff_ij * theta_c * (3 + 6 * k_eff_ij * r_ij_hat + 2 * anp.pi**2 * theta_c**2 * (4 * k_eff_ij * r_ij_hat - 1)) + \
                        3 * (2 * k_eff_ij * r_ij_hat + 1) * (1 + 2 * k_eff_ij * r_ij_hat + 8 * anp.pi**2 * k_eff_ij * r_ij_hat * theta_c**2) * dtheta_c_dr
        outside_brackets = self.fc["a"][0] / (3 * (2 * k_eff_ij * r_ij_hat + 1)**4)
        ddelta_p_dr = anp.nan_to_num(outside_brackets * inside_brackets)

        # Preparing variables for vectorized computation
        theta_ij_hat   = theta_ij_hat[:, :, None]
        r_ij_hat       = r_ij_hat[:, :, None]
        theta_c        = theta_c[:, :, None]
        dtheta_c_dr    = dtheta_c_dr[:, :, None]
        dtheta_c_dkeff = dtheta_c_dkeff[:, :, None]
        k_eff_ij       = k_eff_ij[:, :, None]
        dkeff_dr       = dkeff_dr[:, :, None]
        m = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, 1:]
        a = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, 1:]
        b = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, 1:]

        # Precompute trigonometric terms
        two_pi_m_theta_ij = 2 * anp.pi * m * theta_ij_hat
        two_pi_m_theta_c  = 2 * anp.pi * m * theta_c

        cos_theta_ij = anp.cos(two_pi_m_theta_ij)
        sin_theta_ij = anp.sin(two_pi_m_theta_ij)
        cos_theta_c  = anp.cos(two_pi_m_theta_c)
        sin_theta_c  = anp.sin(two_pi_m_theta_c)

        # Precompute common terms
        two_k_r_plus_one = 2 * k_eff_ij * r_ij_hat + 1
        m_squared = m**2
        k_r = k_eff_ij * r_ij_hat

        # Higher order components of the gradient with respect to r_ij_hat - Equation 42
        A = 1 / (anp.pi * m**3 * two_k_r_plus_one**4)
        B = a * cos_theta_ij + b * sin_theta_ij
        C = -4 * k_eff_ij * sin_theta_c * (
                1 + m_squared + 2 * k_r * (m_squared - 2) + 2 * anp.pi**2 * m_squared * theta_c**2 * (4 * k_r - 1))
        D = 2 * anp.pi * m * cos_theta_c * (
                4 * k_eff_ij * theta_c * (1 - 4 * k_r) +
                m_squared * two_k_r_plus_one * (1 + 2 * k_r + 8 * anp.pi**2 * k_r * theta_c**2) * dtheta_c_dr)

        # Sum over all m-Fourier terms
        ddelta_p_dr += anp.sum(anp.nan_to_num(A * B * (C + D)), axis=2)
        # -----------------

        # -----------------
        # ddelta_p/dk_eff for the TurbOPark-specific EXPLICIT component of the gradient
        # Zero frequency component of the gradient with respect to k_eff.
        inside_brackets = 1 + 8 * anp.pi**2 * k_eff_ij * r_ij_hat * theta_c**2 / (3 * two_k_r_plus_one)
        inside_square_brackets = 8 * anp.pi**2 * r_ij_hat * theta_c**2 / (3 * two_k_r_plus_one**2) - 4 * r_ij_hat / two_k_r_plus_one * inside_brackets
        ddelta_dk_eff = self.fc["a"][0] * theta_c / two_k_r_plus_one**2 * inside_square_brackets

        # Higher order components of the gradient with respect to k_eff.
        Tm = ((two_pi_m_theta_c)**2 - 2) * sin_theta_c + 4 * anp.pi * m * theta_c * cos_theta_c

        A = (a * cos_theta_ij + b * sin_theta_ij) / (anp.pi * m * two_k_r_plus_one**2)
        B = 2 * r_ij_hat / (m_squared * two_k_r_plus_one**2) * Tm
        C = 4 * r_ij_hat / two_k_r_plus_one
        D = sin_theta_c + 2 * k_eff_ij * r_ij_hat / (m_squared * two_k_r_plus_one) * Tm

        ddelta_dk_eff = ddelta_dk_eff.squeeze() + anp.sum(anp.nan_to_num(A * (B - C * D)), axis=2)
        # -----------------

        # -----------------
        # ddelta_p/dtheta_c for the TurbOPark-specific IMPLICIT component of the gradient
        # Zero frequency component of ddelta_p/dtheta_c
        ddelta_p_dtheta_c = self.fc["a"][0] / two_k_r_plus_one**2 * (
            (1 + 8 * anp.pi**2 * k_r * theta_c**2 / (3 * two_k_r_plus_one)) +
            theta_c * (16 * anp.pi**2 * k_r * theta_c / (3 * two_k_r_plus_one))
        )

        # Higher order components of ddelta_p/dtheta_c.
        A = (a * cos_theta_ij + b * sin_theta_ij) / (anp.pi * m * two_k_r_plus_one**2)
        ddelta_p_dtheta_c = ddelta_p_dtheta_c.squeeze() + anp.sum(anp.nan_to_num(
            A * 2 * anp.pi * m * cos_theta_c * (1 + (2 * k_r / (m_squared * two_k_r_plus_one)) * two_pi_m_theta_c**2)
        ), axis=2)
        # -----------------

        # Addition of all components to obtain ddelta_p/dr_ij_hat for the TurbOPark model:
        ddelta_p_dr += (ddelta_dk_eff + ddelta_p_dtheta_c * dtheta_c_dkeff.squeeze()) * dkeff_dr.squeeze()

        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # DERIVATIVE OF WAKE LOSS COMPONENT (delta_p) WITH RESPECT TO THE ANGLE (theta_ij_hat)

        # Higher order derivatives with respect to theta_ij_hat - Equation 40.
        A = 2 / two_k_r_plus_one**2 * (b * cos_theta_ij - a * sin_theta_ij)
        B = sin_theta_c + \
            2 * k_r / (m_squared * two_k_r_plus_one) * (
                (two_pi_m_theta_c**2 - 2) * sin_theta_c + 4 * anp.pi * m * theta_c * cos_theta_c
            )

        ddelta_p_dtheta = anp.sum(anp.nan_to_num(A * B), axis=2)
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # OBTAINING THE GRADIENTS IN CARTESIAN COORDINATES
        # Obtaining free stream power and wake deficits
        p_hat = self.p_hat
        delta_p = self._calculate_delta_p(x, y)

        multiplier = (p_hat - delta_p)**2

        # Derivatives with respect to x and y coordinates - Equations 38 and 39 decomposed
        term_x = anp.nan_to_num(ddelta_p_dr * xij / (r_ij_hat[:, :, 0]) - ddelta_p_dtheta * yij / (2 * anp.pi * r_ij_hat[:, :, 0]**2))
        term_y = anp.nan_to_num(ddelta_p_dr * yij / (r_ij_hat[:, :, 0]) + ddelta_p_dtheta * xij / (2 * anp.pi * r_ij_hat[:, :, 0]**2))

        # Derivatives to account for the movement of the individual windTurbines - Equation 33
        dF_dx = anp.zeros((len(x), 1))
        dF_dy = anp.zeros((len(x), 1))

        # Applying partial derivative with respect to the movement of each individual wind turbine g - Equation 33
        for i in range(len(dF_dx)):
            grad_mask = anp.zeros_like(r_ij_hat[:, :, 0])
            grad_mask[i, :] = -1.   # Equivalent to dxij_dxg
            grad_mask[:, i] =  1.   # Equivalent to dyij_dyg

            # Dimensionless derivative - Equations 38 and 39
            dF_dx[i] = -3 * anp.sum(multiplier * anp.sum(term_x * grad_mask, axis=1))
            dF_dy[i] = -3 * anp.sum(multiplier * anp.sum(term_y * grad_mask, axis=1))
        # ---------------------------------------------------------------------------

        # Return dimensions to gradients
        daep_dx = (dF_dx * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9
        daep_dy = (dF_dy * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9

        if wrt_arg == ['x', 'y']:
            return daep_dx.flatten(), daep_dy.flatten()
        elif wrt_arg == ['x']:
            return daep_dx.flatten()
        elif wrt_arg == ['y']:
            return daep_dy.flatten()