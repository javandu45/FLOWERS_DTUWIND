from FLOWERS.flowers_model import FLOWERS_model

from autograd import numpy as anp

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
            Recommended values are 5-15 modes when using 360 wind directions, default is 10.
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
            Array of length len(x) containing the AEP gradients with respect to the x-coordinates
        daep_dy : np.array
            Array of length len(x) containing the AEP gradients with respect to the y-coordinates
        
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
            grad_mask[i,:] = -1.  
            grad_mask[:,i] = 1.   

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
