from FLOWERS.flowers_model import FLOWERS_model

from autograd import numpy as anp

class gaussian_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - Gaussian Bastankhah Porte-Agel wake model.

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
            Recommended values are 5-15 modes when using 360 wind directions, default is 10.
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

        # Convert to amplitude-phase representation, as in original FLOWERS paper
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

        x = anp.array(x)
        y = anp.array(y)

        # Relative position between turbines i and j (same convention as aep_i)
        xij = (x[:, None] - x[None, :])
        yij = (y[:, None] - y[None, :])

        # Transform to polar coordinates (same convention as aep_i)
        r_ij = anp.sqrt(xij**2 + yij**2)
        theta_ij = anp.arctan2(xij, yij) - anp.pi

        # Adapt theta_ij to wrap the angle into the range [-pi, pi] - m(theta)
        theta_ij = anp.mod(theta_ij + anp.pi, 2 * anp.pi) - anp.pi

        r_ij = anp.where(r_ij == 0, 1e-10, r_ij)  # To avoid division by zero on the diagonal

        # ---------------------------------------------------------------------------
        # PER-PAIR GAUSSIAN WAKE SCALARS AND THEIR DISTANCE DERIVATIVES
        # Gaussian wake width sigma and angular wake width r_a
        sigma = self.k * r_ij + self.epsilon 
        r_a = sigma / r_ij 

        # Centreline deficit g, decomposed as g = 1 - sqrt
        inside_sqrt = anp.where(anp.abs(r_ij) < self.lim, 1, 1 - self.CT / (8 * sigma**2))
        sqrt = anp.sqrt(inside_sqrt)         
        g = 1 - sqrt

        # Derivatives of the per-pair scalars with respect to r_ij, used later
        dr_a = -self.epsilon / r_ij**2
        dg = anp.where(anp.abs(r_ij) < self.lim, 0.0, -self.CT * self.k / (8 * sigma**3 * sqrt))
        # ---------------------------------------------------------------------------

        # Preparing variables for vectorized computation
        theta_ij = theta_ij[:, :, None]
        r_a = r_a[:, :, None]
        dr_a = dr_a[:, :, None]
        g = g[:, :, None]
        dg = dg[:, :, None]
        t = anp.array(self.fc["m"])[anp.newaxis, anp.newaxis, :]
        A = anp.array(self.fc["a"])[anp.newaxis, anp.newaxis, :]
        PHI = anp.array(self.fc["b"])[anp.newaxis, anp.newaxis, :]

        # Gaussian attenuation factors E1 (alpha = 1) and E2 (alpha = 2)
        E1 = anp.exp((-t**2 * r_a**2)/2)
        E2 = anp.exp((-t**2 * r_a**2)/4)

        # ---------------------------------------------------------------------------
        # RADIAL KERNEL K_t AND ITS DISTANCE DERIVATIVE dK_t/dr
        # K_t - Equation G.4 (the same -3*I1 + 3*I2 kernel evaluated in aep_i)
        K = -3 * anp.sqrt(2 * anp.pi) * g * r_a * E1 + 3 * anp.sqrt(anp.pi) * g**2 * r_a * E2

        # dK_t/dr - Equation G.14 (product rule on g, r_a, E1, E2; no theta_c, no k_eff feedback)
        dK = -3 * anp.sqrt(2 * anp.pi) * E1 * (r_a * dg + g * dr_a * (1 - t**2 * r_a**2)) \
             + 3 * anp.sqrt(anp.pi) * g * E2 * (2 * r_a * dg + g * dr_a * (1 - 0.5 * t**2 * r_a**2))
        # ---------------------------------------------------------------------------

        # Trigonometric phase
        phase = t * theta_ij + PHI

        # ---------------------------------------------------------------------------
        # DERIVATIVE OF PER-PAIR WAKE LOSS (W_ij) WITH RESPECT TO r_ij AND theta_ij
        # With respect to r_ij
        dW_dr = anp.sum(A * anp.cos(phase) * dK, axis=-1)
        # With respect to theta_ij
        dW_dtheta = -anp.sum(A * t * anp.sin(phase) * K, axis=-1)
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # OBTAINING THE GRADIENTS IN CARTESIAN COORDINATES
        term_x = anp.nan_to_num(dW_dr * xij / r_ij + dW_dtheta * yij / r_ij**2)
        term_y = anp.nan_to_num(dW_dr * yij / r_ij - dW_dtheta * xij / r_ij**2)

        # Derivatives to account for the movement of the individual windTurbines
        dF_dx = anp.zeros((len(x), 1))
        dF_dy = anp.zeros((len(x), 1))

        # Applying partial derivative with respect to the movement of each individual wind turbine p
        for p in range(len(dF_dx)):
            grad_mask = anp.zeros_like(r_ij)
            grad_mask[p, :] += 1.   
            grad_mask[:, p] -= 1.   

            dF_dx[p] = anp.sum(term_x * grad_mask)
            dF_dy[p] = anp.sum(term_y * grad_mask)
        # ---------------------------------------------------------------------------

        # Return dimensions to gradients (same scaling as aep_i)
        daep_dx = dF_dx * 0.5 * 8760 * self.rho * RotorDiameter**2 / 4 * anp.pi / 1e9
        daep_dy = dF_dy * 0.5 * 8760 * self.rho * RotorDiameter**2 / 4 * anp.pi / 1e9

        if wrt_arg == ['x', 'y']:
            return daep_dx.flatten(), daep_dy.flatten()
        elif wrt_arg == ['x']:
            return daep_dx.flatten()
        elif wrt_arg == ['y']:
            return daep_dy.flatten()
