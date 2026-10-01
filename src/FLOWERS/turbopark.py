from FLOWERS.flowers_model import FLOWERS_model

from autograd import numpy as anp


class TurbOPark_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - Nygaard TurbOPark wake model.

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
            Recommended values are 5-15 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3
        ti : float
            Turbulence intensity, default is 0.1
            
        """

        k = None # k is not a parameter to be defined in TurbOPark

        super().__init__(site, windTurbines, k, n_terms, ws_cutout, rho)

        # Universal thrust coefficient
        self.CT = self._universal_ct()

        # Fourier coefficients
        fourier_function = self.cp**(1/3) * self.avg_ws_norm * (1-anp.sqrt(1-self.ct)) * self.freqs
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
        dkeff_dr = self.A * self.I0 / (2 * r_ij_hat) * (common / (anp.sqrt(common**2 + 1) + 1) + 1 / common) - k_eff_ij / r_ij_hat

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
