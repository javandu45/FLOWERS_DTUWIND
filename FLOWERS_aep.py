import autograd.numpy as anp
from autograd import grad
from scipy.special import gamma
from matplotlib import pyplot as plt


class NOJ_flowers():

    """
    FLOW Estimation and Rose Superposition - NO Jensen wake model.

    FLOWERS is an AEP model which efficiently computes a wind farm's AEP. It uses numerous simplifications
    and assumptions, but most importantly a Fourier transform turning discrete components, functions of wind
    direction, into a continuous function. From this, it  solves the integral analytically, rather than 
    numerically, also enabling the computation of analytical gradients.

    Based on "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization"
    https://doi.org/10.1002/we.2954
    """

    def __init__(self, site, WindTurbine, k = 0.04, n_terms=10, ws_cutout=25, rho=1.225):

        """
        Model initialization. Given its approach, FLOWERS presents the following modelling limitations:
            - All wind turbines throughout the wind farm must be of the same type.
            - Wind conditions must remain uniform in the wind farm area, atmospheric homogeneity.

        Parameters
        ----------
        Site : Site
            Site Object (UniformWeibullSite)
        windTurbine : WindTurbine
            WindTurbine object representing the wake generating wind turbines
        k : float
            Wake expansion coefficient, for NO Jensen wake model, default is 0.04
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3

        """

        self.WindTurbine = WindTurbine
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
        ideal_power = 1/2 * self.rho * self.WindTurbine.diameter()**2/4 * anp.pi * self.avg_ws**3
        self.cp = self.WindTurbine.power(self.avg_ws)/ideal_power
        self.ct = self.WindTurbine.ct(self.avg_ws)

        # Getting fourier coefficients
        self.fc = self.fourier_coefficients()

        # Free stream AEP component for a single turbine - Equation 18
        self.p_hat = self.calculate_p_hat()
        

    def calculate_p_hat(self):

        """
        Free stream AEP component for a single turbine - Equation 18

        Returns
        -------
        p_hat : float

        """

        # Free stream AEP component for a single turbine - Equation 18
        p_hat = anp.sum(self.cp**(1/3) * self.avg_ws_norm * self.freqs)

        return float(p_hat)
    
    
    def calculate_delta_p(self, x, y):

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

        D = self.WindTurbine.diameter()

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
        theta_c = anp.nan_to_num(anp.arctan((1/(2*r_ij_hat + epsilon) + self.k * anp.sqrt(1 + self.k**2 - (1/(2*r_ij_hat))**2)) / 
                          (-self.k/(2*r_ij_hat + epsilon) + anp.sqrt(1 + self.k**2 - (1/(2*r_ij_hat + epsilon))**2)) / (2 * anp.pi)))

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
    

    def calculate_AEP(self, x, y):

        """
        Computes the wind farm's AEP using FLOWERS model, defined as the summation of the contribution from each
        turbine. Such contribution is the result of substracting all wake interactions experienced by turbine i
        (delta_p) from the free stream AEP component for a turbine (p_hat)

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

        # Free stream AEP component for a single wind turbine - Equation 18
        p_hat = self.p_hat

        # Wake loss component - Equation 28
        delta_p = self.calculate_delta_p(x, y)

        # Sum over all turbines (i), dimensionless AEP - Right hand side of equation 16
        aep = sum((p_hat - delta_p)**3)

        # Final AEP computation - Solving for AEP in equation 16
        aep = aep * 8760 * anp.pi/8 * self.rho * self.WindTurbine.diameter()**2 * (self.ws_cutout**3)/1e9

        return aep
    
    
    def AEP_per_turbine(self, x, y):

        """
        Computes the AEP contribution from each turbine, which is the result of substracting all wake interactions
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
        delta_p = self.calculate_delta_p(x, y)

        # AEP contribution from each turbine i (freestream AEP - wakes from all turbines on turbine i)
        aep_i = (p_hat - delta_p)**3

        # Giving back dimensions

        aep_i = aep_i * 8760 * anp.pi/8 * self.rho * self.WindTurbine.diameter()**2 * (self.ws_cutout**3)/1e9

        return aep_i
    

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

        aep_turbines = self.AEP_per_turbine(x, y)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c=aep_turbines, cmap='viridis', s=100)
        plt.colorbar(label='AEP per Turbine (GWh)')
        plt.title('AEP per Turbine using NOJ Flowers Model')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.show()
    
    
    def fourier_coefficients(self):

        """
        Obtain Fourier coefficients to transform discrete components, which are a function of wind direction,
        into continuous form.

        Returns
        -------
        fc : dict
            a : Array of length n_terms containing Fourier coefficient a
            b : Array of length n_terms containing Fourier coefficient b
            m : Array of length n_terms containing the coefficient index

        """

        n_terms = self.n_terms

        # Terms that are only a function of the discrete wind direction - Equation 23
        c_fourier = self.cp**(1/3) * self.avg_ws_norm * (1-anp.sqrt(1-self.ct)) * self.freqs

        # Fourier coefficients: a_0, a_m, b_m
        coeffs = 2 * anp.fft.rfft(c_fourier)
        a = coeffs.real
        b = -coeffs.imag

        # Use only the first n_terms
        if n_terms > 0 and n_terms < len(a):
            a = a[:n_terms]
            b = b[:n_terms]
            m = anp.arange(n_terms)
            fc = {"a": a, "b": b, "m": m}

            return fc

        else:
            raise ValueError("n_terms should be between 0 and n_wd/2")
        
    
    def calculate_gradients(self, coords, method="Autograd"):

        """
        Compute the AEP gradients with respect to the turbine positions x and y.

        Parameters
        ----------
        coords : array_like
            A 2D array of shape (2, n) where:
            - The first row contains x-coordinates
            - The second row contains y-coordinates
        method : {"Autograd", "Exact"}
            The method to use for computation.
            - "Autograd": Uses automatic differentiation using the autograd package.
            - "Exact": Uses exact analytical gradients.

        Returns
        -------
        daep_dx : np.array
            Array of length n containing the AEP gradients with respect to the x-coordinates
        daep_dy : np.array
            Array of length n containing the AEP gradients with respect to the y-coordinates
        
        """

        if method not in ["Autograd", "Exact"]:
            raise ValueError("method must be either 'Autograd' or 'Exact'.")

        coords = anp.asarray(coords, dtype=float)

        # Unpacking coordinates
        x, y = coords

        # Gradients using autograd package
        if method == "Autograd":

            # Gradient with respect to both coordinates at the same time to save computational time
            aep_wrapped = lambda coords: self.calculate_AEP(coords[0], coords[1])

            gradient = grad(aep_wrapped)
            daep_dx, daep_dy = gradient(coords)

            return daep_dx, daep_dy

        # Exact gradients based on FLOWERS AEP paper
        elif method == "Exact":

            RotorDiameter = self.WindTurbine.diameter()

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
            delta_p = self.calculate_delta_p(x, y)

            multiplier = (p_hat - delta_p)**2

            # Derivatives with respect to x and y coordinates - Equations 38 and 39 decomposed
            term_x = anp.nan_to_num(ddelta_p_dr * xij / (r_ij_hat[:,:,0]) - ddelta_p_dtheta * yij / (2 * anp.pi * r_ij_hat[:,:,0]**2))
            term_y = anp.nan_to_num(ddelta_p_dr * yij / (r_ij_hat[:,:,0]) + ddelta_p_dtheta * xij / (2 * anp.pi * r_ij_hat[:,:,0]**2))

            # Derivatives to account for the movement of the individual WindTurbine - Equation 33
            dF_dx = anp.zeros((len(x), 1))
            dF_dy = anp.zeros((len(x), 1))

            # Applying partial derivative with respect to the movement of each individual wind WindTurbine g - Equation 33
            for i in range(len(dF_dx)):
                grad_mask = anp.zeros_like(r_ij_hat[:,:,0])
                grad_mask[i,:] = -1.  # Equivalent to dxij_dxg
                grad_mask[:,i] = 1.   # Equivalent to dyij_dyg

                # Dimensionaless derivative - Equations 38 and 39
                dF_dx[i] = -3*anp.sum(multiplier *  anp.sum(term_x * grad_mask, axis=1))
                dF_dy[i] = -3*anp.sum(multiplier * anp.sum(term_y * grad_mask, axis=1))
            # ---------------------------------------------------------------------------

            daep_dx = (dF_dx * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9
            daep_dy = (dF_dy * anp.pi / 8 * 8760 * self.rho * RotorDiameter * self.ws_cutout**3) / 1e9

            return daep_dx.flatten(), daep_dy.flatten()


class gaussian_flowers():

    """
    FLOW Estimation and Rose Superposition - Gaussian Bastankhah wake model.

    FLOWERS is an AEP model which efficiently computes a wind farm's AEP. It uses numerous simplifications
    and assumptions, but most importantly a Fourier transform turning discrete components, functions of wind
    direction, into a continuous function. From this, it solves the integral analytically, rather than
    numerically. For the moment, analytical gradients have not yet been derived.

    Based on "Gaussian FLOWERS: Wind-rose-based analytical integration of Gaussian wake model for extremely fast
    AEP estimation"
    https://doi.org/10.1063/5.0245886
    """

    def __init__(self, WindTurbine, site, k=0.03, n_terms=10, ws_cutout=25, rho=1.225):

        """
        Model initialization. Given its approach, FLOWERS presents the following modelling limitations:
            - All wind turbines throughout the wind farm must be of the same type.
            - Wind conditions must remain uniform in the wind farm area, atmospheric homogeneity.

        Parameters
        ----------
        Site : Site
            Site Object (UniformWeibullSite)
        windTurbine : WindTurbine
            WindTurbine object representing the wake generating wind turbines
        k : float
            Wake expansion coefficient, for Gaussian Bastankhah wake model, default is 0.03
        n_terms : int
            Number of Fourier modes to compute for the Fourier Transform. Maximum number is n_wd/2 + 1.
            A higher amount of modes results in an increased accuracy, but also a higher computational cost.
            Recommended values are 10-20 modes when using 360 wind directions, default is 10.
        ws_cutout : float
            Wind turbine cut-out wind speed, default is 25 m/s
        rho : float
            Air density, default is 1.225 kg/m3    

        """

        self.WindTurbine = WindTurbine
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
        ideal_power = 1/2 * self.rho * self.WindTurbine.diameter()**2/4 * anp.pi * self.avg_ws**3
        self.cp = self.WindTurbine.power(self.avg_ws)/ideal_power
        self.ct = self.WindTurbine.ct(self.avg_ws)

        # Getting fourier coefficients
        self.fc = self.fourier_coefficients()

        # Getting universal thrust coefficient
        self.CT = self.universal_ct()

        # Gaussian wake parameters, based on universal thrust coefficient
        self.beta = (1 + anp.sqrt(1 - self.CT)) / (2 * anp.sqrt(1 - self.CT))
        self.epsilon = 0.2 * anp.sqrt(self.beta)

        # Limit to avoid NaNs in gaussian wake deficit
        self.lim = 1 / self.k * anp.sqrt(self.CT / 8) - self.epsilon


    def fourier_coefficients(self):

        """
        Obtain Fourier coefficients to transform discrete components, which are a function of wind direction,
        into continuous form.

        Returns
        -------
        fc : dict
            a : Array of length n_terms containing Fourier amplitude
            b : Array of length n_terms containing Fourier phase angle
            m : Array of length n_terms containing the mode's index

        """

        # Inspiration taken from author's code

        # Terms that are only a function of the discrete wind direction - Equation 11
        c_fourier = len(self.freqs) / (2*anp.pi) * self.freqs * self.cp * self.avg_ws**3

        # Fourier coefficients: a_0, a_m, b_m
        coeffs = anp.fft.rfft(c_fourier)/len(c_fourier)
        a_0 = coeffs.real[0]
        a = 2*coeffs.real[1:]
        b = -2*coeffs.imag[1:]

        # Prepare for Fourier amplitude phase representation
        A = anp.sqrt(a ** 2 + b ** 2)
        B = -anp.arctan2(b, a)

        n_terms = self.n_terms

        # Use only the first n_terms
        if n_terms > 0 and n_terms <= len(a):
            a = anp.concatenate((anp.array([a_0]), A[:n_terms-1])) # Add a_0 at the beginning
            b = anp.concatenate((anp.array([0]), B[:n_terms-1]))   # Add 0 as a_0's phase angle
            m = anp.arange(n_terms)
            fc = {"a": a, "b": b, "m": m}

            return fc

        else:
            raise ValueError("n_terms should be between 0 and n_wd/2")
            

    def universal_ct(self):

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
    

    def calculate_AEP(self, x, y):

        """
        Computes the wind farm's AEP using FLOWERS model

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

        # AEP contribution from each turbine i
        aep_i = self.AEP_per_turbine(x, y)

        # Sum over all turbines (i)
        aep = anp.sum(aep_i)

        return aep
    

    def calculate_gradients(self, coords):

        """
        Compute the AEP gradients with respect to the turbine positions x and y. For the moment, gradients
        can only be obtained using automatic differentiation.

        Parameters
        ----------
        coords : array_like
            A 2D array of shape (2, n) where:
            - The first row contains x-coordinates
            - The second row contains y-coordinatess.

        Returns
        -------
        daep_dx : np.array
            Array of length n containing the AEP gradients with respect to the x-coordinates
        daep_dy : np.array
            Array of length n containing the AEP gradients with respect to the y-coordinates
        
        """

        # Gradient with respect to both coordinates at the same time to save computational time
        aep_wrapped = lambda coords: self.calculate_AEP(coords[0], coords[1])

        gradient = grad(aep_wrapped)
        daep_dx, daep_dy = gradient(coords)

        return daep_dx, daep_dy
    

    def AEP_per_turbine(self, x, y):

        """
        Computes the AEP contribution from each turbine

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
        aep_i = anp.sum(- 3 * I1 + 3 * I2, axis=-1)

        # Sum over all turbines j
        aep_i = anp.sum(aep_i, axis=-1)

        # Sum over all turbines i (adding I0 component), dimensionless AEP
        aep_i = aep_i + I0

        aep_i = aep_i * 0.5 * 8760 * self.rho * self.WindTurbine.diameter()**2/4 * anp.pi / 1e9

        return aep_i
    
    
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

        aep_turbines = self.AEP_per_turbine(x, y)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c=aep_turbines, cmap='viridis', s=100)
        plt.colorbar(label='AEP per Turbine (GWh)')
        plt.title('AEP per Turbine using Gaussian FLOWERS Model')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.show()

