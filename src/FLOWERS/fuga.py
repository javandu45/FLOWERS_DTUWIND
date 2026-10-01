from FLOWERS.flowers_model import FLOWERS_model

from autograd import numpy as anp
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.special import jv
import numpy as np


class fuga_flowers(FLOWERS_model):

    """
    FLOW Estimation and Rose Superposition - Fuga wake model.

    Supports two AEP computation methods via the `method` parameter:

    - "linear" (default): NOJ-FLOWERS style linear deficit summation — deficits from all upstream
      turbines j are summed linearly into delta_p(i) and AEP_i = (p_hat - delta_p)^3.

    - "binomial": Gaussian-FLOWERS style binomial expansion — (1 - sum_j d_j)^3 is expanded to
      quadratic order: I0 - 3*sum_j I1_j + 3*sum_j I2_j, using both the linear and squared
      wake-shape Fourier coefficients.

    Both methods support source="lut" (real-space Cartesian Fuga LUT) or source="flut"
    (angular-spectral fLUT, processed via angular DFT and Hankel transform).
    """

    def __init__(self, site, windTurbines, n_terms=10, ws_cutout=25, rho=1.225,
                 lut_file=None, source="lut", method="linear",
                 r_min=None, r_max=None, n_r=1000, n_alpha=360, stability_conditions_prob=None,
                 induction=True):

        """
        Model initialization. Given its approach, FLOWERS presents the following modelling limitations:
            - All wind turbines throughout the wind farm must be of the same type.
            - Wind conditions must remain uniform in the wind farm area, atmospheric homogeneity.

        For more information regarding the derivation, please refer to the docs/Derivations.md

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
        lut_file : str
            Path to the Fuga LUT (source="lut") or fLUT (source="flut") file.
        source : {"lut", "flut"}
            Whether to use the real-space Fuga LUT ("lut") or the angular-spectral fLUT ("flut").
        method : {"linear", "binomial"}
            The method to use for AEP computation.
        r_min, r_max : float, optional
            r-grid bounds [rotor diameters] for source="flut" (defaults: 0.1, 200).
        n_r : int
            r-grid size for source="flut" (default 1000).
        n_alpha : int
            Angular resolution for ring-sampling or angular DFT (default 360).
        stability_conditions_prob : array-like, optional
            Probability of each stability condition (length n_wd). To be used only when atmoshperic stability
            is being considered. (EXPERIMENTAL)
        induction : bool
            Whether to include induction effects in the wake shape. If False, the wake is truncated at
            theta = +-pi/2 (default True).

        """

        super().__init__(site=site, windTurbines=windTurbines, k=None,
                         n_terms=n_terms, ws_cutout=ws_cutout, rho=rho)

        if source not in ("lut", "flut"):
            raise ValueError("source must be 'lut' or 'flut'")
        if method not in ("linear", "binomial"):
            raise ValueError("method must be 'linear' or 'binomial'")

        self.method = method
        self.source = source
        self.induction = induction

        # Load the LUT and derive r-grid parameters
        if source == "lut":

            self.lut = xr.open_dataset(lut_file)
            D = windTurbines.diameter()
            # Downstream distance step to create the grid for the fourier moments
            dx = np.diff(self.lut.coords['x'].values)[0] / D
            # Maximum and minimum downstream distances for the fourier moments
            if r_min is None:   
                r_min = 0.1
            if r_max is None:
                r_max = max(self.lut.coords['x'].values) / D
            n_r = int(r_max / dx)

        elif source == "flut":
            self.flut = xr.open_dataset(lut_file)
            # Maximum and minimum downstream distances for the fourier moments
            if r_min is None:
                r_min = 0.1
            if r_max is None:
                r_max = 200

        # If the stability conditions are not provided, diregard this parameter, otherwise compute the 
        # weighted stability factor w_s to be used in the Fourier series of the wind-rose
        if stability_conditions_prob is None:
            stability_conditions_prob = np.ones(len(self.freqs))
            w_s = np.ones(len(self.freqs))
        else:
            self.stability_conditions_prob = stability_conditions_prob
            w_s = self._weighted_stability()

        # Wind-rose Fourier series (method-specific)
        if method == "linear":
            fourier_function = self.avg_ws_norm * self.cp**(1/3) * self.freqs * self.ct * w_s
            self.fc = self._fourier_coefficients(fourier_function)
            self.p_hat = self._calculate_p_hat()
        else:
            fourier_function = self.cp * self.avg_ws**3 * self.freqs * w_s
            self.fc0 = self._fourier_coefficients(fourier_function)
            self.fc1 = self._fourier_coefficients(fourier_function * self.ct)
            self.fc2 = self._fourier_coefficients(fourier_function * self.ct**2)
            # Halve a[0] to match the real Fourier series convention
            self.fc0["a"][0] /= 2
            self.fc1["a"][0] /= 2
            self.fc2["a"][0] /= 2

        # Build the Fourier moments of the wake shape (method- and source-specific)
        self._build_moments(r_min=r_min, r_max=r_max, n_r=n_r, n_alpha=n_alpha)


    def _weighted_stability(self):

        """
        Compute the weighted stability factor w_s to be used in the Fourier series of the wind-rose, given the
        Weibull parameters of the site and the stability conditions probabilities.

        The stability conditions probabilities are the probability of the specific conditions being used in this
        case for each wind speed considered. This function then computes the energy weighted probability of this 
        stability condition based on the speed probability for each wind direction. In other words, the energy 
        weighted probability of the studied atmoshperic stability for each wind direction.
        """

        ws = self.site.default_ws
        wds = np.arange(0, 361, 1)
        ideal_power = 1/2 * self.rho * self.windTurbines.diameter()**2/4 * anp.pi * ws**3
        cps = self.windTurbines.power(ws)/ideal_power

        k_weibull = self.site.ds.Weibull_k.values
        A_weibull = self.site.ds.Weibull_A.values
        p_ws = np.zeros((len(ws), len(wds)))
        w_s = np.zeros((len(wds), 1))
        for i, _ in enumerate(wds):

            # Wind speed probability for each wind direction, using the weibull parameters
            p_ws[:, i] = k_weibull[i] / A_weibull[i] * (ws / A_weibull[i])**(k_weibull[i]-1) * np.exp(-(ws/A_weibull[i])**k_weibull[i])

            # Numerator: energy coming for this wind direction for this stability condition
            top = cps * ws **3 * self.stability_conditions_prob * p_ws[:, i]

            # Denominator: total energy coming for this wind direction for all stability conditions
            bottom = cps * ws**3 * p_ws[:, i]
            w_s[i] = np.sum(top)/np.sum(bottom)

        return w_s.flatten()


# ----------------------------
# AUXILIARY FUNCTIONS
# ----------------------------

    @staticmethod
    def _polar2cartesian(r, theta):

        x = r * anp.cos(theta)
        y = r * anp.sin(theta)

        return x, y

    @staticmethod
    def _nearest_idx_sorted(sorted_arr, query):

        """Nearest index in a sorted 1D array for each query point (O(n log n))."""

        idx = np.searchsorted(sorted_arr, query)
        idx = np.clip(idx, 1, len(sorted_arr) - 1)

        left = sorted_arr[idx - 1]
        right = sorted_arr[idx]
        idx -= (query - left) < (right - query)
        return idx

    @staticmethod
    def _make_linear_interp(r_grid, y):

        """Autograd-differentiable piecewise-linear interpolator over r_grid."""

        r_grid = anp.asarray(r_grid)
        y = anp.asarray(y)
        def interp(x):
            x = anp.asarray(x)
            raw = anp.asarray(x._value if hasattr(x, "_value") else x)
            idx = anp.clip(anp.searchsorted(r_grid, raw, side="right") - 1, 0, len(r_grid) - 2)
            r0 = r_grid[idx]
            r1 = r_grid[idx + 1]
            y0 = y[idx]
            y1 = y[idx + 1]
            val = y0 + (x - r0) / (r1 - r0) * (y1 - y0)
            val = anp.where(x < r_grid[0],  y[0],  val)
            val = anp.where(x > r_grid[-1], y[-1], val)
            return val
        return interp

    @staticmethod
    def _compute_interpolator_gradients(r_grid, y):

        """Piecewise-constant slope of the linear interpolator (dy/dr)."""

        r_grid = anp.asarray(r_grid)
        y = anp.asarray(y)
        def interp_grad(x):
            x = anp.asarray(x)
            raw = anp.asarray(x._value if hasattr(x, "_value") else x)
            idx = anp.clip(anp.searchsorted(r_grid, raw, side="right") - 1, 0, len(r_grid) - 2)
            dy = (y[idx + 1] - y[idx]) / (r_grid[idx + 1] - r_grid[idx])
            dy = anp.where(x < r_grid[0],  0.0, dy)
            dy = anp.where(x > r_grid[-1], 0.0, dy)
            return dy
        return interp_grad

# ----------------------------
# FUNCTIONS TO BUILD LUT OF MODES TO RECONSTRUCT WAKE SHAPE
# ----------------------------

    def _LUT(self, r, theta):

        """Nearest-neighbour lookup of the real-space Fuga LUT at polar (r [D], theta)."""

        D = self.windTurbines.diameter()
        r = anp.asarray(r)
        theta = anp.asarray(theta)
        orig_shape = r.shape

        x, y = self._polar2cartesian(r, theta)
        x = np.asarray(x).ravel() * D
        y = np.abs(np.asarray(y).ravel()) * D  # LUT stores only one symmetric half in y

        try:
            y_coords = self.lut.coords['y'].values
            x_coords = self.lut.coords['x'].values
        except Exception:
            y_coords = self.lut['y'].values
            x_coords = self.lut['x'].values

        idx_x = self._nearest_idx_sorted(x_coords, x)
        idx_y = self._nearest_idx_sorted(y_coords, y)

        du = self.lut.UL.values[0, idx_x, idx_y]

        # Values outisde look-up table are set to zero
        outside = (x < x_coords[0]) | (x > x_coords[-1]) | (y > y_coords[-1])
        du = np.where(outside, 0.0, du).reshape(orig_shape)
        
        # du (downstream wind speed) = du_from_luts * c_t * ws * zeta0_factor
        # c_t * ws is already included in Fourier coefficients, they are functions of
        # wind direction
        scale = self._zeta0_factor()

        if not self.induction:
            theta_wrapped = (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
            du = np.where(np.abs(theta_wrapped) <= np.pi/2, du, 0.0)

        return -du*scale


    def _zeta0_factor(self):

        """
        Obtain zeta0_factor to recover dimensional wake deficits for conditions different
        to neutral.
        
        Taken directly from PyFuga documentation.
        """

        zeta0 = self.lut.attrs['zeta0'].astype(float)
        zhub = self.windTurbines.hub_height()
        z0 = float(self.lut['z0'].values)

        invL = zeta0 / z0
        ams = 5 

        def psim(zeta):
            if zeta0 >= 0:
                return -ams * zeta
            else:
                amu = -19.3
                aux2 = np.sqrt(1 + amu * zeta)
                aux = np.sqrt(aux2)
                return np.pi / 2 - 2 * np.arctan(aux) + np.log((1 + aux) ** 2 * (1 + aux2) / 8)

        return 1 - (psim(zhub * invL) - psim(zeta0)) / np.log(zhub / z0)


    def _build_interpolators(self, r_grid, coeff_array):

        """Build value and slope interpolators for every column of coeff_array (n_r × M)."""

        M = coeff_array.shape[1]
        splines = [self._make_linear_interp(r_grid, coeff_array[:, m]) for m in range(M)]
        slopes = [self._compute_interpolator_gradients(r_grid, coeff_array[:, m]) for m in range(M)]
        return splines, slopes


    def _hankel_transform(self, r_min, r_max, n_r, n_alpha, n_k_oversample=4000):

        """
        Compute angular Fourier moments of the wake via the Hankel transform
        """
        M     = self.n_terms
        D     = self.windTurbines.diameter()

        # --------------------------------------------------------------------
        # FLUT angular modes
        # --------------------------------------------------------------------
        ds = self.flut
        z0 = float(ds['z0'].values)
        kz0 = ds.coords['kz0'].values.astype(float)
        beta_q = ds.coords['beta'].values.astype(float)
        UL = ds['UL'].values[:, :, 0, :]
        F = UL[..., 0] + 1j * UL[..., 1]
        k = kz0 / z0

        beta_fine = np.linspace(beta_q.min(), beta_q.max(), n_alpha)
        Re_fine   = np.empty((n_alpha, len(k)))
        Im_fine   = np.empty((n_alpha, len(k)))
        for j in range(len(k)):
            Re_fine[:, j] = CubicSpline(beta_q, F[:, j].real)(beta_fine)
            Im_fine[:, j] = CubicSpline(beta_q, F[:, j].imag)(beta_fine)

        ghat = np.zeros((M, len(k)), dtype=complex)
        for n in range(M):
            cosnb = np.cos(n * beta_fine)[:, None]
            if n % 2 == 0:
                ghat[n] = (2/np.pi)  * np.trapezoid(Re_fine * cosnb, beta_fine, axis=0)
            else:
                ghat[n] = (2j/np.pi) * np.trapezoid(Im_fine * cosnb, beta_fine, axis=0)

        # ---------------------------------------------------------------------
        # Oversample in log-k and Hankel transform to obtain c_n(r)
        # ---------------------------------------------------------------------
        pos = k > 0
        kl = np.log(k[pos])
        kl_f = np.linspace(kl.min(), kl.max(), n_k_oversample)
        k_fine = np.exp(kl_f)
        ghat_fine = np.empty((ghat.shape[0], n_k_oversample), dtype=complex)
        for n in range(ghat.shape[0]):
            ghat_fine[n] = (CubicSpline(kl, ghat[n, pos].real)(kl_f)
                           + 1j * CubicSpline(kl, ghat[n, pos].imag)(kl_f))

        # ---------------------------------------------------------------------
        # Scale factor for the Hankel transform (log-law height factor)
        # ---------------------------------------------------------------------
        kappa = 0.4
        z_hub = self.windTurbines.hub_height()
        z0 = float(self.flut['z0'].values)
        scale = -np.log(z_hub / z0) / kappa

        r_grid = anp.linspace(r_min, r_max, n_r)
        r_phys = np.asarray(r_grid) * D
        kf = k_fine[:, None]

        c_n = np.empty((M, n_r), dtype=complex)
        for m in range(M):
            Jm = jv(m, kf * r_phys[None, :])
            integrand = ghat_fine[m][:, None] * Jm * kf
            c_n[m] = (1j**m / (2 * np.pi)) * np.trapezoid(integrand, k_fine, axis=0)
        c_n *= scale

        imag_frac = np.max(np.abs(c_n.imag)) / (np.max(np.abs(c_n.real)) + 1e-30)
        if imag_frac > 1e-3:
            import warnings
            warnings.warn(f"[fLUT] Im(c_n)/Re(c_n) ~ {imag_frac:.2e}; check "
                          "hub-height slice / scaling / mirror symmetry.")
        return r_grid, c_n


    def _build_moments(self, r_min, r_max, n_r, n_alpha):
        """
        Build and store the Fourier-coefficient interpolators used by aep_i and the
        exact gradient.
        """
        # Obtain the Fourier coefficients of the wake shape (A, B) on a radial grid
        if self.source == "lut":
            r_grid = anp.linspace(r_min, r_max, n_r)
            alpha = anp.linspace(0, 2 * anp.pi, n_alpha, endpoint=False)
            R = anp.broadcast_to(r_grid[:, None], (n_r, n_alpha))
            ALPHA = anp.broadcast_to(alpha[None, :],  (n_r, n_alpha))
            G = np.asarray(self._LUT(R, ALPHA))
            fc_raw = self._fourier_coefficients(G / n_alpha)
            A = fc_raw["a"]          # (n_r, M)
            B = fc_raw["b"]          # (n_r, M)
        else:
            r_grid, c_n = self._hankel_transform(r_min, r_max, n_r, n_alpha)
            A = 2 * c_n.real.T     # (n_r, M)
            B = -2 * c_n.imag.T    # (n_r, M)

        self._r_grid = r_grid

        # Obtain interpolating functions for the Fourier moments of the wake shape,
        # and their slopes which are needed for the gradients
        if self.method == "linear":
            self._A_interpolator, self._A_slopes = self._build_interpolators(r_grid, A)
            self._B_interpolator, self._B_slopes = self._build_interpolators(r_grid, B)
        elif self.method == "binomial":
            c_n_real = A.T / 2           
            M = c_n_real.shape[0]
            def c(i):
                i = abs(i)
                return c_n_real[i] if i < M else np.zeros(c_n_real.shape[1])
            q_n = np.empty_like(c_n_real)
            for n in range(M):
                q_n[n] = sum(c(k) * c(n - k) for k in range(-(M - 1), M))
            p1 = A                  
            p2 = (2 * q_n).T   
            self._p1_interpolator, self._p1_slopes = self._build_interpolators(r_grid, p1)
            self._p2_interpolator, self._p2_slopes = self._build_interpolators(r_grid, p2)


# ----------------------------
# FUNCTIONS TO EVALUATE MODES AND ITS GRADIENTS
# ----------------------------

    def _eval_mode_interpolators(self, splines, r):

        """Evaluate a list of splines at r, returning shape r.shape + (len(splines),)."""

        r_flat = anp.asarray(r).ravel()
        result = anp.stack([spl(r_flat) for spl in splines], axis=-1)
        return result.reshape(r.shape + (len(splines),))


    def _eval_moments(self, r, for_gradient=False):

        """
        Evaluate the wake-shape Fourier moments at separations r, dispatching on self.method:
          - "linear"  : returns (A, B) or (A, B, dA, dB)   when for_gradient=True
          - "binomial": returns (p1, p2) or (p1, p2, dp1, dp2) when for_gradient=True
        """

        if self.method == "linear":
            c1  = self._eval_mode_interpolators(self._A_interpolator, r)
            c2  = self._eval_mode_interpolators(self._B_interpolator, r)
            if for_gradient:
                dc1 = self._eval_mode_interpolators(self._A_slopes, r)
                dc2 = self._eval_mode_interpolators(self._B_slopes, r)
        else:
            c1  = self._eval_mode_interpolators(self._p1_interpolator, r)
            c2  = self._eval_mode_interpolators(self._p2_interpolator, r)
            if for_gradient:
                dc1 = self._eval_mode_interpolators(self._p1_slopes, r)
                dc2 = self._eval_mode_interpolators(self._p2_slopes, r)
        return (c1, c2, dc1, dc2) if for_gradient else (c1, c2)

# ----------------------------
# FUNCTIONS TO COMPUTE AEP WITH FLOWERS
# ----------------------------

    def _calculate_p_hat(self):

        p_hat = float(anp.sum(self.cp**(1/3) * self.avg_ws_norm * self.freqs))

        return p_hat


    def aep_i(self, x, y):

        if self.method == "linear":
            return self._aep_i_linear(x, y)
        elif self.method == "binomial": 
            return self._aep_i_binomial(x, y)


    def _aep_i_linear(self, x, y):

        # Compute the wake-loss integral delta_p for the linear summation method
        delta_p = self._calculate_delta_p(x, y)

        # Compute the AEP for each turbine using the linear deficit summation method
        aep_turbine = (self.p_hat - delta_p)**3

        # Return dimensions to AEP in GWh
        aep_turbine = self.ws_cutout**3 * aep_turbine * 8760 * anp.pi/8 * self.rho * self.windTurbines.diameter()**2 / 1e9

        return aep_turbine


    def _calculate_delta_p(self, x, y):

        """
        Dimensionless wake-loss integral for the linear summation method, using the convolution
        theorem to fold the wind-rose fourier coefficients (self.fc) with the wake-shape Fourier 
        Coefficients (A, B).
        """

        D = self.windTurbines.diameter()
        x = anp.array(x);  y = anp.array(y)

        # Relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None]) / D
        yij = (y[None, :] - y[:, None]) / D

        # Transform to polar coordiantes
        r_ij_hat = anp.where(anp.sqrt(xij**2 + yij**2) == 0, 1e-6, anp.sqrt(xij**2 + yij**2))
        theta_ij = anp.arctan2(yij, xij)/(2 * anp.pi)

        # Evaluate the wake-shape Fourier coefficients (A, B) at the relative distances r_ij_hat
        A, B = self._eval_moments(r_ij_hat)

        # Prepare variables for vectorized computation of delta_p
        a_f0 = self.fc["a"][0]       
        m = anp.array(self.fc["m"])[None, None, 1:]
        a_f = anp.array(self.fc["a"])[None, None, 1:]
        b_f = anp.array(self.fc["b"])[None, None, 1:]
        theta = theta_ij[:, :, None]
        A0 = A[:, :, 0]
        A = A[:, :, 1:]
        B = B[:, :, 1:]

        # Precompute to speed up computation
        cos_t = anp.cos(2*np.pi * m * theta)
        sin_t = anp.sin(2*np.pi * m * theta)

        # Zero order term
        zero_t = 1 / 4 * A0 * a_f0

        # Higher order terms
        higher_t = (a_f*A - b_f*B) * cos_t + (a_f*B + b_f*A) * sin_t

        # Add and sum over fourier terms
        delta_p = zero_t + 1/2 * anp.sum(higher_t, axis=2)

        # Remove self-contribution of each turbine to its own wake loss
        off_diagonal = 1 - anp.eye(len(x))
        delta_p = delta_p * off_diagonal

        # Sum over all upstream turbines j to get the total wake loss for each turbine i
        delta_p = anp.sum(delta_p, axis=1)

        return delta_p


    def _aep_i_binomial(self, x, y):

        """
        Gaussian-style binomial expansion:
            AEP_i = (I0 + sum_j pi*(-3*I1_ij + 3*I2_ij)) x dimensional constant
        where I1, I2 carry the linear and squared wake-shape fourier coefficients weighted
        by the direction-varying thrust (fc1, fc2).
        
        """

        D = self.windTurbines.diameter()
        x = anp.array(x)
        y = anp.array(y)

        # Relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None]) / D
        yij = (y[None, :] - y[:, None]) / D

        # Transform to polar coordiantes
        r_ij_hat = anp.where(anp.sqrt(xij**2 + yij**2) == 0, 1.0, anp.sqrt(xij**2 + yij**2))
        theta_ij = anp.arctan2(yij, xij)

        # Evaluate the wake-shape Fourier coefficients (p1, p2) at the relative distances r_ij_hat
        p1, p2 = self._eval_moments(r_ij_hat)

        # Prepare variables for vectorized computation 
        t = anp.arange(self.n_terms)[None, None, :]
        theta = theta_ij[:, :, None]
        a1 = anp.array(self.fc1["a"])[None, None, :]
        b1 = anp.array(self.fc1["b"])[None, None, :]
        a2 = anp.array(self.fc2["a"])[None, None, :]
        b2 = anp.array(self.fc2["b"])[None, None, :]

        # Precompute to speed up computation
        cos_mt = anp.cos(t * theta)
        sin_mt = anp.sin(t * theta)

        # Compute the three terms of the binomial expansion
        I0 = self.fc0["a"][0]
        I1 = 1/2 * (a1 * cos_mt + b1 * sin_mt) * p1
        I2 = 1/2 * (a2 * cos_mt + b2 * sin_mt) * p2

        # Add terms and sum over fourier terms
        aep_turbine = anp.sum(-3 * I1 + 3 * I2, axis=2)

        # Remove self-contribution of each turbine to its own wake loss
        off_diagonal = 1 - anp.eye(len(x))
        aep_turbine = aep_turbine * off_diagonal

        # Add zero order term, and sum over all upstream turbines j to get the total wake loss for each turbine i
        aep_turbine = I0 + anp.sum(aep_turbine, axis=1)

        # Return dimensions to AEP in GWh
        aep_turbine = aep_turbine * 0.5 * 8760 * self.rho * D**2 / 4 * anp.pi / 1e9

        return aep_turbine

# ----------------------------
# FUNCTIONS TO COMPUTE EXACT GRADIENT OF AEP WITH FLOWERS
# ----------------------------

    def _aep_gradient_exact(self, x, y, wrt_arg=['x', 'y']):

        if self.method == "linear":
            return self._gradient_linear(x, y, wrt_arg)
        elif self.method == "binomial":
            return self._gradient_binomial(x, y, wrt_arg)


    def _gradient_linear(self, x, y, wrt_arg):

        D = self.windTurbines.diameter()
        x = anp.array(x)
        y = anp.array(y)

        # Compute the relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None]) / D
        yij = (y[None, :] - y[:, None]) / D

        # Transform to polar coordinates
        r_ij_hat = anp.where(anp.sqrt(xij**2 + yij**2) == 0, 1e-6, anp.sqrt(xij**2 + yij**2))
        theta_ij = anp.arctan2(yij, xij)

        # Evaluate the wake-shape Fourier coefficients (A, B) and their slopes (dA, dB)
        # at the relative distances r_ij_hat
        A, B, dA, dB = self._eval_moments(r_ij_hat, for_gradient=True)

        # Prepare variables for vectorized computation of the gradients
        a_f0 = self.fc["a"][0]
        m = anp.array(self.fc["m"])[None, None, 1:]
        a_f = anp.array(self.fc["a"])[None, None, 1:]
        b_f = anp.array(self.fc["b"])[None, None, 1:]
        theta = theta_ij[:, :, None]
        A0  = A[:, :, 0]
        A  = A[:, :, 1:]
        B  = B[:, :, 1:]
        dA0 = dA[:, :, 0]
        dA = dA[:, :, 1:]
        dB = dB[:, :, 1:]

        # Precompute to speed up computation
        cos_t = anp.cos(m * theta)
        sin_t = anp.sin(m * theta)

        # ddelta_p / dr
        ddp_dr   = (a_f*dA - b_f*dB)*cos_t + (a_f*dB + b_f*dA)*sin_t
        ddp_dr   = 1/4 * a_f0 * dA0 + 1/2 * anp.sum(ddp_dr, axis=2)

        # ddelta_p / dtheta
        ddp_dth  = 1/2 * m * (-(a_f*A - b_f*B)*sin_t + (a_f*B + b_f*A)*cos_t)
        ddp_dth  = anp.sum(ddp_dth, axis=2)

        # OBTAINING THE GRADIENTS IN CARTESIAN COORDINATES
        # Obtain wake loss component
        delta_p    = self._calculate_delta_p(x, y)
        multiplier = (self.p_hat - delta_p)**2

        # Derivatives with respect to x and y coordinates
        term_x = anp.nan_to_num(ddp_dr * xij / r_ij_hat - ddp_dth * yij / r_ij_hat**2)
        term_y = anp.nan_to_num(ddp_dr * yij / r_ij_hat + ddp_dth * xij / r_ij_hat**2)

        # Derivatives to account for the movement of the individual windTurbines
        dF_dx = anp.zeros((len(x), 1))
        dF_dy = anp.zeros((len(x), 1))

        # Applying partial derivative with respect to the movement of each individual wind windTurbines g
        for i in range(len(x)):
            mask = anp.zeros_like(r_ij_hat)
            mask[i, :] = -1.
            mask[:, i] = 1.

            # Dimensionless derivative
            dF_dx[i] = -3 * anp.sum(multiplier * anp.sum(term_x * mask, axis=1))
            dF_dy[i] = -3 * anp.sum(multiplier * anp.sum(term_y * mask, axis=1))

        # Return dimensions to gradients in GWh/m
        scale   = anp.pi / 8 * 8760 * self.rho * D * self.ws_cutout**3 / 1e9
        daep_dx = dF_dx * scale
        daep_dy = dF_dy * scale

        if wrt_arg == ['x', 'y']:
            return daep_dx.flatten(), daep_dy.flatten()
        elif wrt_arg == ['x']:
            return daep_dx.flatten()
        elif wrt_arg == ['y']:
            return daep_dy.flatten()


    def _gradient_binomial(self, x, y, wrt_arg):

        D = self.windTurbines.diameter()
        x = anp.array(x)
        y = anp.array(y)

        # Relative normalized position between turbines i and j
        xij = (x[None, :] - x[:, None]) / D
        yij = (y[None, :] - y[:, None]) / D

        # Transform to polar coordinates
        r_ij_hat = anp.where(anp.sqrt(xij**2 + yij**2) == 0, 1.0, anp.sqrt(xij**2 + yij**2))
        theta_ij = anp.arctan2(yij, xij)

        # Evaluate the wake-shape Fourier coefficients (p1, p2) and their slopes (dp1, dp2)
        # at the relative distances r_ij_hat
        p1, p2, dp1, dp2 = self._eval_moments(r_ij_hat, for_gradient=True)

        # Prepare variables for vectorized computation of the gradients
        t = anp.arange(self.n_terms)[None, None, :]
        theta = theta_ij[:, :, None]
        a1 = anp.array(self.fc1["a"])[None, None, :]
        b1 = anp.array(self.fc1["b"])[None, None, :]
        a2 = anp.array(self.fc2["a"])[None, None, :]
        b2 = anp.array(self.fc2["b"])[None, None, :]

        # Precompute to speed up computation
        cos_mt = anp.cos(t * theta)
        sin_mt = anp.sin(t * theta)

        dK_dr  = anp.pi * anp.sum(
            -3 * (a1*cos_mt + b1*sin_mt) * dp1
            + 3 * (a2*cos_mt + b2*sin_mt) * dp2, axis=2)

        dK_dth = anp.pi * anp.sum(
            -3 * t * (-a1*sin_mt + b1*cos_mt) * p1
            + 3 * t * (-a2*sin_mt + b2*cos_mt) * p2, axis=2)


        # OBTAINING THE GRADIENTS IN CARTESIAN COORDINATES
        # Derivatives with respect to x and y coordinates
        term_x = anp.nan_to_num(dK_dr * xij / r_ij_hat - dK_dth * yij / r_ij_hat**2)
        term_y = anp.nan_to_num(dK_dr * yij / r_ij_hat + dK_dth * xij / r_ij_hat**2)

        # Derivatives to account for the movement of the individual windTurbines
        dF_dx = anp.zeros((len(x), 1))
        dF_dy = anp.zeros((len(x), 1))

        # Applying partial derivative with respect to the movement of each individual wind windTurbines g
        for p in range(len(x)):
            mask = anp.zeros_like(r_ij_hat)
            mask[p, :] = -1.;  mask[:, p] = 1.

            # Dimensionless derivative
            dF_dx[p] = anp.sum(term_x * mask)
            dF_dy[p] = anp.sum(term_y * mask)

        # Return dimesions to gradients
        scale = 0.5 * 8760 * self.rho * D / 4 * anp.pi / 1e9
        daep_dx = dF_dx * scale
        daep_dy = dF_dy * scale

        if wrt_arg == ['x', 'y']:
            return daep_dx.flatten(), daep_dy.flatten()
        elif wrt_arg == ['x']:
            return daep_dx.flatten()
        elif wrt_arg == ['y']:
            return daep_dy.flatten()

