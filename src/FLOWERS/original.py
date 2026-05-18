# from utils import *

# FLOWERS

# Michael LoCascio

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

class FlowersInterface():
    """
    Flowers is a high-level user interface to the FLOWERS AEP model.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        num_terms (int, optional): number of Fourier modes
        k (float, optional): wake expansion rate
        turbine (str, optional): turbine type:
                - 'nrel_5MW' (default)

    """

    ###########################################################################
    # Initialization tools
    ###########################################################################

    def __init__(self, wind_rose, layout_x, layout_y, num_terms=0, k=0.05, turbine=None):

        self.wind_rose = wind_rose
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.k = k

        if turbine is None or turbine == 'nrel_5MW':
            self.turbine = 'nrel_5MW'
            self.D = 126.
            self.U = 25.0
        
        self._fourier_coefficients(num_terms=num_terms)

    
    def reinitialize(self, wind_rose=None, layout_x=None, layout_y=None, num_terms=None, k=None):

        if wind_rose is not None:
            self.wind_rose = wind_rose
            self._fourier_coefficients(num_terms=num_terms)
        
        if num_terms is not None:
            self._fourier_coefficients(num_terms=num_terms)
        
        if layout_x is not None:
            self.layout_x = layout_x
        
        if layout_y is not None:
            self.layout_y = layout_y
        
        if k is not None:
            self.k = k
    
    ###########################################################################
    # User functions
    ###########################################################################

    def get_layout(self):
        return self.layout_x, self.layout_y
    
    def get_wind_rose(self):
        return self.wind_rose
    
    def get_num_modes(self):
        return len(self.fs)
    
    def calculate_aep(self, gradient=False):
        """
        Compute farm AEP (and Cartesian gradients) for the given layout and wind rose.
        
        Returns:
            aep (float): farm AEP [Wh]
            gradient (numpy.array(float)): (dAEP/dx, dAEP/dy) for each turbine [Wh/m]
        """
        
        # Power component from freestream
        u0 = self.fs["c"]

        # Normalize and reshape relative positions into symmetric 2D array
        xx = (self.layout_x - np.reshape(self.layout_x,(-1,1)))/self.D
        yy = (self.layout_y - np.reshape(self.layout_y,(-1,1)))/self.D

        # Convert to normalized polar coordinates
        R = np.sqrt(xx**2 + yy**2)
        THETA = np.arctan2(yy,xx) / (2 * np.pi)

        # Set up mask for rotor swept area
        mask_area = np.array(R <= 0.5, dtype=int)
        mask_val = self.fs["c"]

        # Critical polar angle of wake edge (as a function of distance from turbine)
        theta_c = np.arctan(
            (1 / (2*R) + self.k * np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            / (-self.k / (2*R) + np.sqrt(1 + self.k**2 - (2*R)**(-2)))
            ) / (2 * np.pi)
        theta_c = np.nan_to_num(theta_c)

        # Contribution from zero-frequency Fourier mode
        du = self.fs["a"][0] * theta_c / (2 * self.k * R + 1)**2 * (
            1 + (8 * np.pi**2 * theta_c**2 * self.k * R) / (3 * (2 * self.k * R + 1)))
        
        # Initialize gradient and calculate zero-frequency modes
        if gradient == True:
            grad = np.zeros((len(self.layout_x),2))

            # Change in theta_c wrt radius
            dtdr = (-1 / (4 * np.pi * R**2 * np.sqrt(self.k**2 - (2*R)**(-2) + 1)))
            dtdr = np.nan_to_num(dtdr)

            # Zero-frequency mode of change in power deficit wrt radius
            dpdr = (-4 * self.fs["a"][0] * self.k * theta_c * (3 + 6 * self.k * R + 2 * np.pi**2 * (4 * self.k * R - 1) * theta_c**2) + 
                    3 * self.fs["a"][0] * (1 + 2 * self.k * R) * (1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr) / (
                3 * (1 + 2*self.k*R)**4)
            
        # Reshape variables for vectorized calculations
        m = np.arange(1, len(self.fs["b"]))
        a = np.swapaxes(np.tile(np.expand_dims(self.fs["a"][1:], axis=(1,2)),np.shape(R.T)),0,2)
        b = np.swapaxes(np.tile(np.expand_dims(self.fs["b"][1:], axis=(1,2)),np.shape(R.T)),0,2)
        R = np.tile(np.expand_dims(R, axis=2),len(m))
        THETA = np.tile(np.expand_dims(THETA, axis=2),len(m))
        theta_c = np.tile(np.expand_dims(theta_c, axis=2),len(m))

        # Vectorized contribution of higher Fourier modes
        du += np.sum((1 / (np.pi * m * (2 * self.k * R + 1)**2) * (
            a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) * (
                np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                    ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)

        if gradient==True:
            dtdr = np.tile(np.expand_dims(dtdr, axis=2),len(m))
            
            # Higher Fourier modes of change in power deficit wrt angle
            dpdt = np.sum((2 / (2 * self.k * R + 1)**2 * (
                b * np.cos(2 * np.pi * m * THETA) - a * np.sin(2 * np.pi * m * THETA)) * (
                    np.sin(2 * np.pi * m * theta_c) + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                        ((2 * np.pi * theta_c * m)**2 - 2) * np.sin(2 * np.pi * m * theta_c) + 4*np.pi*m*theta_c*np.cos(2 * np.pi * m * theta_c)))), axis=2)

            # Higher Fourier modes of change in power deficit wrt radius
            dpdr += np.sum(((a * np.cos(2 * np.pi * m * THETA) + b * np.sin(2 * np.pi * m * THETA)) / (np.pi * m**3 * (2 * self.k * R + 1)**4) * (
                -4 * self.k * np.sin(2 * np.pi * m * theta_c) * (1 + m**2 + 2 * self.k * R * (m**2 - 2) + 2 * np.pi**2 * m**2 * (4 * self.k * R - 1) * theta_c**2) + 
                2 * np.pi * m * np.cos(2 * np.pi * m * theta_c) * (4 * self.k * (1 - 4 * self.k * R) * theta_c + m**2 * (2 * self.k * R + 1) * (
                1 + 2 * self.k * R + 8 * np.pi**2 * self.k * R * theta_c**2) * dtdr))), axis=2)

        # Apply mask for points within rotor radius
        du = du * (1 - mask_area) + mask_val * mask_area
        np.fill_diagonal(du, 0.)
        
        # Sum power for each turbine
        du = np.sum(du, axis=1)
        aep = np.sum((u0 - du)**3)

        aep *= np.pi / 8 * 1.225 * self.D**2 * self.U**3 * 8760

        # Complete gradient calculation
        if gradient==True:
            dx = xx/np.sqrt(xx**2+yy**2)*dpdr + -yy/(2*np.pi*(xx**2+yy**2))*dpdt
            dy = yy/np.sqrt(xx**2+yy**2)*dpdr + xx/(2*np.pi*(xx**2+yy**2))*dpdt

            dx = np.nan_to_num(dx)
            dy = np.nan_to_num(dy)
            
            coeff = (u0 - du)**2
            for i in range(len(grad)):
                # Isolate gradient to turbine 'i'
                grad_mask = np.zeros_like(xx)
                grad_mask[i,:] = -1.
                grad_mask[:,i] = 1.

                grad[i,0] = np.sum(coeff*np.sum(dx*grad_mask,axis=1)) 
                grad[i,1] = np.sum(coeff*np.sum(dy*grad_mask,axis=1))

            grad *= -3 * np.pi / 8 * 1.225 * self.D * self.U**3 * 8760

            return aep, grad
            # return dpdr, dpdt
        
        else:
            return aep

    ###########################################################################
    # Private functions
    ###########################################################################

    def _fourier_coefficients(self, num_terms=0):
        """
        Compute the Fourier series expansion coefficients from the wind rose.
        Modifies the Flowers interface in place to add a Fourier coefficients
        dataframe:
            fs (pandas:dataframe): Fourier coefficients used to expand the wind rose:
                - 'a_free': real coefficients of freestream component
                - 'a_wake': real coefficients of wake component
                - 'b_wake': imaginary coefficients of wake component

        Args:
            num_terms (int, optional): the number of Fourier modes to save in the range
                [1, floor(num_wind_directions/2)]
        
        """


        # Resample wind rose for average wind speed per wind direction
        wr = self.wind_rose.copy()
        # wr = resample_average_ws_by_wd(wr)

        # # Transform wind direction to polar angle 
        # wr["wd"] = np.remainder(450 - wr.wd, 360)
        # wr.sort_values("wd", inplace=True)
        # wr.loc[len(wr)] = wr.iloc[0]
        # wr.freq_val /= np.sum(wr.freq_val)

        # Normalize wind speed by cut-out speed
        wr["ws"] /= self.U

        # # Look up thrust and power coefficients for each wind direction bin
        # ct = ct_lookup(wr["ws"],self.turbine)
        # cp = cp_lookup(wr["ws"],self.turbine)

        # Using same turbine as new FLOWERS code, required due to numerical precision
        tur = nrel_5MW()
        ct = tur.ct(wr["ws"] * self.U)
        ideal_power = 0.5 * 1.225 * (126.0**2 / 4) * np.pi * (wr["ws"] * self.U)**3
        cp = tur.power(wr["ws"] * self.U)/ ideal_power

        # Average freestream term
        c = np.sum(cp**(1/3) * wr["ws"] * wr["freq_val"])

        # Fourier expansion of wake deficit term
        c1 = cp**(1/3) * (1 - np.sqrt(1 - ct)) * wr["ws"] * wr["freq_val"]

        c1ft = 2 * np.fft.rfft(c1)
        a =  c1ft.real
        b = -c1ft.imag

        # Truncate Fourier series to specified number of modes
        if num_terms > 0 and num_terms <= len(a):
            a = a[0:num_terms]
            b = b[0:num_terms]

        # Compile Fourier coefficients
        self.fs = {'a': a, 'b': b, 'c': c}


###########################################################################
# Turbine parameter tables
###########################################################################

def ct_lookup(u, turbine_type, ct=None):
    """
    Look-up table for thrust coefficient of the NREL 5 MW turbine.

    Args:
        u (float): normalized inflow wind speed
    
    Returns:
        ct (float): thrust coefficient
    
    """

    if ct != None:
        ct_table = np.array([0.0, 0.0, ct, ct, 0.0, 0.0])
        u_table = 1/25. * np.array([0.0, 2.0, 2.5, 25.01, 25.02, 50.])
    elif turbine_type == 'nrel_5MW':
        ct_table = np.array([0.0, 0.0, 0.0, 0.99, 0.99, 0.97373036, 0.92826162, 0.89210543,
        0.86100905, 0.835423, 0.81237673, 0.79225789, 0.77584769, 0.7629228, 0.76156073,
        0.76261984, 0.76169723, 0.75232027, 0.74026851, 0.72987175, 0.70701647, 0.54054532,
        0.45509459, 0.39343381, 0.34250785, 0.30487242, 0.27164979, 0.24361964, 0.21973831,
        0.19918151, 0.18131868, 0.16537679, 0.15103727, 0.13998636, 0.1289037, 0.11970413,
        0.11087113, 0.10339901, 0.09617888, 0.09009926, 0.08395078, 0.0791188, 0.07448356,
        0.07050731, 0.06684119, 0.06345518, 0.06032267, 0.05741999, 0.05472609, 0.0, 0.0])
        u_table = 1 / 25. * np.array([0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
        14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.01, 25.02, 50.0])
    
    return np.interp(u, u_table, ct_table)

def cp_lookup(u, turbine_type, cp=None):
    """
    Look-up table for power coefficient of the NREL 5 MW turbine.

    Args:
        u (float): normalized inflow wind speed
    
    Returns:
        cp (float): power coefficient
    
    """
    if cp != None:
        cp_table = np.array([0.0, 0.0, cp, cp, 0.0, 0.0])
        u_table = 1/25. * np.array([0.0, 2.0, 2.5, 25.01, 25.02, 50.])
    elif turbine_type == 'nrel_5MW':
        cp_table = np.array([0.0, 0.0, 0.0, 0.178085, 0.289075, 0.349022, 0.384728,
        0.406059, 0.420228, 0.428823, 0.433873, 0.436223, 0.436845, 0.436575, 0.436511,
        0.436561, 0.436517, 0.435903, 0.434673, 0.433230, 0.430466, 0.378869, 0.335199,
        0.297991, 0.266092, 0.238588, 0.214748, 0.193981, 0.175808, 0.159835, 0.145741,
        0.133256, 0.122157, 0.112257, 0.103399, 0.095449, 0.088294, 0.081836, 0.075993,
        0.070692, 0.065875, 0.061484, 0.057476, 0.053809, 0.050447, 0.047358, 0.044518,
        0.041900, 0.039483, 0.0, 0.0])
        u_table = 1 / 25. * np.array([0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
        14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.01, 25.02, 50.0])

    return np.interp(u, u_table, cp_table)

def resample_average_ws_by_wd(df):
        """
        Calculate the mean wind speed for each wind direction bin
        and resample the wind rose. (Copied from FLORIS)

    Args:
        df (pandas.DataFrame): Wind rose DataFrame containing the following
            columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Wind speed bin center values (m/s).
            - 'freq_val': The frequency of occurance of the
                wind conditions in the other columns.

    Returns:
        New wind rose DataFrame containing the following columns:
            - 'wd': Wind direction bin center values (deg).
            - 'ws': Resampled average wind speed values (m/s).
            - 'freq_val': The resampled frequency of occurance of the
                wind conditions in the other columns.
                
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        ws_avg = []

        for val in df.wd.unique():
            ws_avg.append(
                np.array(
                    df.loc[df["wd"] == val]["ws"] * df.loc[df["wd"] == val]["freq_val"]
                ).sum()
                / df.loc[df["wd"] == val]["freq_val"].sum()
            )

        # Regroup
        df = df.groupby("wd").sum()

        df["ws"] = ws_avg

        # Reset the index
        df = df.reset_index()

        # Set to float
        df["ws"] = df.ws.astype(float)
        df["wd"] = df.wd.astype(float)

        return df


def load_wind_rose(idx):
    """
    Load a locally-stored wind rose saved to a pickle file.
    See show_wind_roses.py to visualize all wind rose options.

    Args:
        idx (int): index of desired wind rose
    
    Returns:
        df (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction

    """

    print("Generating wind rose.")
    file_name = 'wr' + str(idx) + '.p'
    df = pd.read_pickle(file_name)
    
    return df


def discrete_layout(n_turb=0, D=126.0, min_dist=3.0, idx=None, spacing=False):
    """
    Generate a random wind farm layout within the specified boundaries.
    Minimum spacing between turbines is 2D.

    Args:
        boundaries (list(tuple)): boundary vertices in the form
            [(x0,y0), (x1,y1), ... , (xN,yN)]
        n_turb (int): number of turbines
        D (float): rotor diameter [m]
        min_dist (float): enforced minimum spacing between turbine centers
            normalized by rotor diameter
        idx (int, optional): random number generator seed
    
    Args:
        xx (np.array): x-positions of each turbine
        yy (np.array): y-positions of each turbine

    """

    print("Generating wind farm layout.")
    
    if n_turb <= 0:
        raise ValueError("Must supply number of turbines.")

    # Initialize RNG and containers
    if idx != None:
        np.random.seed(idx)

    xx = np.zeros(n_turb)
    yy = np.zeros(n_turb)

    # Indices of discrete grid
    s = n_turb + 1
    x_idx = np.random.randint(0,s,n_turb)
    y_idx = np.random.randint(0,s,n_turb)
    pts = [(x_idx[i],y_idx[i]) for i in range(n_turb)]
    while len(np.unique(pts)) < len(pts):
        tmp = np.unique(pts)
        new_set = []
        for i in range(n_turb):
            if i not in tmp:
                new_set.append(i)
        x_idx[new_set] = np.random.randint(0,s,len(new_set))
        y_idx[new_set] = np.random.randint(0,s,len(new_set))
        pts = [(x_idx[i],y_idx[i]) for i in range(n_turb)]

    # Check that all combinations of x,y are unique

    xx = np.array(min_dist*D * x_idx)
    yy = np.array(min_dist*D * y_idx)

    if spacing:
        x_rel = (xx - np.reshape(xx,(-1,1)))/D
        y_rel = (yy - np.reshape(yy,(-1,1)))/D
        r_rel = np.sqrt(x_rel**2 + y_rel**2)
        r_rel = np.ma.masked_where(np.eye(len(xx)),r_rel)
        ss = np.mean(np.min(r_rel,-1))
        return xx, yy, ss
    else:
        return xx, yy
    
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine

def nrel_5MW():

    """
    Create a WindTurbine object representing the NREL 5MW turbine using tabular Cp and Ct data.
    """

    diameter = 126.0

    ct_table = np.array([0.0, 0.0, 0.0, 0.99, 0.99, 0.97373036, 0.92826162, 0.89210543,
        0.86100905, 0.835423, 0.81237673, 0.79225789, 0.77584769, 0.7629228, 0.76156073,
        0.76261984, 0.76169723, 0.75232027, 0.74026851, 0.72987175, 0.70701647, 0.54054532,
        0.45509459, 0.39343381, 0.34250785, 0.30487242, 0.27164979, 0.24361964, 0.21973831,
        0.19918151, 0.18131868, 0.16537679, 0.15103727, 0.13998636, 0.1289037, 0.11970413,
        0.11087113, 0.10339901, 0.09617888, 0.09009926, 0.08395078, 0.0791188, 0.07448356,
        0.07050731, 0.06684119, 0.06345518, 0.06032267, 0.05741999, 0.05472609, 0.0, 0.0], dtype=np.float64)
    
    u_table = np.array([0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
        14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.01, 25.02, 50.0], dtype=np.float64)
    
    cp_table = np.array([0.0, 0.0, 0.0, 0.178085, 0.289075, 0.349022, 0.384728,
        0.406059, 0.420228, 0.428823, 0.433873, 0.436223, 0.436845, 0.436575, 0.436511,
        0.436561, 0.436517, 0.435903, 0.434673, 0.433230, 0.430466, 0.378869, 0.335199,
        0.297991, 0.266092, 0.238588, 0.214748, 0.193981, 0.175808, 0.159835, 0.145741,
        0.133256, 0.122157, 0.112257, 0.103399, 0.095449, 0.088294, 0.081836, 0.075993,
        0.070692, 0.065875, 0.061484, 0.057476, 0.053809, 0.050447, 0.047358, 0.044518,
        0.041900, 0.039483, 0.0, 0.0], dtype=np.float64)

    p_table = 0.5 * 1.225 * (diameter**2 / 4) * np.pi * u_table**3 * cp_table / 1e6

    wt = WindTurbine(name="NREL_5MW",
                    diameter=diameter,
                    hub_height=90,
                    powerCtFunction=PowerCtTabular(ws=u_table, power=p_table, power_unit="MW", ct=ct_table))
    
    return wt
    


# ================================================================
# GAUSSIAN FLOWERS
# ================================================================

def ntag_PA(Fourier_coeffs3_PA,
            layout1,layout2,
            turb,
            K,
            wav_Ct,
            u_lim=3,
            RHO=1.225):
    
    """ 
    "No cross Terms Analytical Gaussian (Phase Amplitude)" - The "Gaussian FLOWERS" method implemented 

    Args:
        Fourier_coeffs3_PA (list of np.arrays): tuple of Fourier coefficient (a_0,A_n,Phi_n), such that f(theta)= a_0/2 + (A_n*np.cos(n*theta+Phi_n)
        reconstructs the wind rose Cp(U(theta))*P(theta)*U(theta)**3
        layout1 (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines
        layout2 (nt,2) OR (n_grid_points,2): when layout2 = layout1, Uwt_j      gives the power/wake velocity at the turbine locations. if layout2 = plot_points, (the power result is meaningless) Uwt_j is the wake velocity at the plot points (useful for plotting)
        turb (turbine object): must have: turb.A area attribute, Ct_f and Cp_f methods for interpolating the thrust and power cofficient curve
        K (float): Gaussian wake expansion parameter
        wav_Ct (float): The constant thrust coefficient 
        RHO (float): assumed atmospheric density
        u_lim (float): user defined invalid radius limit. This sets a radius around the turbine where the deficit is zero (useful for plotting)

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        alpha (nt,2) | (plot_points,2) : "energy content" (Cp(U)*P*U**3) of the wind at turbine locations or plot_points
    """
    r_jk,theta_jk = find_relative_coords(layout1,layout2)  #find relative posistions
    theta_jk = theta_jk - np.pi #wake lies opposite
    theta_jk = np.mod(theta_jk + np.pi, 2 * np.pi) - np.pi #fix domain

    #is this necessary?!

    A_n,Phi_n = Fourier_coeffs3_PA
    A_n = A_n[:20]
    Phi_n = Phi_n[:20]
    a_0 = 2*A_n[0] #because A_n[0] = a_0 / 2

    EP = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

    #auxilaries 
    n = np.arange(0,A_n.size,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)


    lim = (np.sqrt(wav_Ct/8)-EP)/K
    lim = np.where(lim<u_lim,u_lim,lim) #pick greater from u_lim and lim
    if np.any((r_jk<lim) & (r_jk != 0)):
        raise ValueError("turbines within the invalid region, this will likely cause erroneously low AEP")
    #if turbine 1 is posistioned adjacent to turbine 2, neither upwind or downwind (""inline with each other, perpendicular to the wind direction"")", if within the r limit, turbine 1 will be waked by turbine 2 - which is not realistic (or atleast not as described by Bastankah 2014)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(wav_Ct/(8*(K*r_jk+EP)**2))))) #careful, even though it uses a small angle approximation, the domain is still restricted exactly (?)
    
    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None]
    #more auxilaries
    fs = A_n*np.cos(n_b*theta_b+Phi_n) #fourier series (including DC!)
    nsigma = sigma_b*n_b

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (np.sum(np.exp(-((nsigma)**2)/(2*a))*(fs),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)

    #alpha is the 'energy' content of the wind
    alpha = (a_0/2)*2*np.pi - 3*term(1) + 3*term(2) #- term(3)
    #print("alpha: {}".format(alpha))
    #(I fully vectorised this and it ran slower ... so I'm sticking with this)
    #If it were vectorised using dimensions sparingly (e.g. don't broadcast everything to 4D (alpha,J,K,N) ) immediately) it might be faster
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*turb.A*RHO*alpha)/(1*10**9)*8760
    else: #farm wake visualisation, power is meaningless
        pow_j = np.nan
    return pow_j,alpha

def find_relative_coords(layout,plot_points):
    #find the r, theta coordinates relative to each turbine
    xt_j,yt_j = layout[:,0],layout[:,1]
    xt_k,yt_k = plot_points[:,0],plot_points[:,1]

    x_jk = xt_k[:, None] - xt_j[None, :]
    y_jk = yt_k[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.arctan2(x_jk,y_jk) #clockwise +ve from +ve y axis

    return r_jk,theta_jk  

import timeit
def adaptive_timeit(func,timed=True):
    # this times func() (can't have any arguments) over ~ 4-8 secs and returns a single-execution run time in seconds
    # (define func using a lambda function with no arguments before hand ... )
    result = func() #get the actual result of the function
    if timed is not True: #don't bother timing
        return result,np.nan

    #find the correct number to take 0.75-1.5 secs
    number = 5  # 5 iterations to start
    while True:
        # Time how long it takes for 'number' iterations
        elapsed_time = timeit.timeit(lambda: func(), number=number)
        if elapsed_time >= 0.75: 
            break
        number *= 2  # Double number of iterations

    # Now use 'repeat' to run the test multiple times
    times = timeit.repeat(lambda: func(), number=number, repeat=5)
    #this should take ~4-8 secs
    return result,min(times)/number  # Return the best time


def si_fm(number):
    # Display a value to 3dp in scientific forma using SI prefix
    # e.g. 1234 is 1.23k

    prefixes = {
        24: 'Y',  # yotta
        21: 'Z',  # zetta
        18: 'E',  # exa
        15: 'P',  # peta
        12: 'T',  # tera
        9: 'G',   # giga
        6: 'M',   # mega
        3: 'k',   # kilo
        0: '',    # (no prefix)
        -3: 'm',  # milli
        -6: 'µ',  # micro
        -9: 'n',  # nano
        -12: 'p', # pico
        -15: 'f', # femto
        -18: 'a', # atto
        -21: 'z', # zepto
        -24: 'y'  # yocto
    }
    # Find the appropriate prefix for the number
    for exp, prefix in prefixes.items():
        if number >= 10 ** exp:
            break
    value = round(number / (10 ** exp), 3)

    # Return the formatted string
    return f"{value}{prefix}"


def simple_Fourier_coeffs(data):   
    # naively fit a Fourier series to data (no normalisation takes place (!))
    # returns both sine/cosine and phase/amplitude coefficients
    # reconstruction uses the formula:
    # a_0/2 + (a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)
    # or 
    # a_0/2 + (A_n*np.cos(n_b*theta_b+Phi_n)
    import scipy.fft
    c = scipy.fft.rfft(data)/np.size(data)
    a_0 = 2*np.real(c[0]) 
    a_n = 2*np.real(c[1:])
    b_n =-2*np.imag(c[1:])
    Fourier_coeffs = a_0,a_n,b_n
    # #convert to phase amplitude form
    A_n = np.sqrt(a_n**2+b_n**2)
    Phi_n = -np.arctan2(b_n,a_n)
    #add the dc term on the front with Phi_0 = 0
    A_n = np.concatenate((np.array((a_0/2,)),A_n))
    Phi_n = np.concatenate((np.array((0,)),Phi_n))
    Fourier_coeffs_PA = A_n,Phi_n #pack
    
    return Fourier_coeffs,Fourier_coeffs_PA

class iea_10MW():
    def __init__(self):
        import numpy as np
        self.Cp = np.array([0.0, 0.0, 0.074, 0.3251, 0.3762, 0.4027, 0.4156, 0.423, 0.4274, 0.4293, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4305, 0.438256, 0.425908, 0.347037, 0.307306, 0.271523, 0.239552, 0.211166, 0.186093, 0.164033, 0.144688, 0.12776, 0.112969, 0.100062, 0.0888, 0.078975, 0.070401, 0.062913, 0.056368, 0.05064, 0.04562, 0.041216, 0.037344, 0.033935, 0.0, 0.0])

        self.Ct = np.array([0.0, 0.0, 0.7701, 0.7701, 0.7763, 0.7824, 0.782, 0.7802, 0.7772, 0.7719, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7675, 0.7651, 0.7587, 0.5056, 0.431, 0.3708, 0.3209, 0.2788, 0.2432, 0.2128, 0.1868, 0.1645, 0.1454, 0.1289, 0.1147, 0.1024, 0.0918, 0.0825, 0.0745, 0.0675, 0.0613, 0.0559, 0.0512, 0.047, 0.0, 0.0])

        self.wind_speed = np.array([0.0, 2.9, 3.0, 4.0, 4.5147, 5.0008, 5.4574, 5.8833, 6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484, 7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337, 8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717, 9.3202, 9.6035, 9.921, 10.272, 10.6557, 10.7577, 11.5177, 11.9941, 12.4994, 13.0324, 13.592, 14.1769, 14.7859, 15.4175, 16.0704, 16.7432, 17.4342, 18.1421, 18.8652, 19.6019, 20.3506, 21.1096, 21.8773, 22.6519, 23.4317, 24.215, 25.01, 25.02, 50.0])

        self.power_table = 0.5 * 1.225 * (198.0**2 / 4) * np.pi * self.wind_speed**3 * self.Cp

        self.Z_h = 119.0 #height
        self.D = 198.0 #diameter
        self.A = np.pi*(self.D/2)**2 #area
        self.U = 25. #cut out speed
        self.name = 'iea_10mw'

    def Cp_f(self,u):
        ideal_power = 0.5 * 1.225 * (198.0**2 / 4) * np.pi * u**3
        cp = np.interp(u,self.wind_speed,self.power_table)/ideal_power
        return cp

    def Ct_f(self,u):
        return np.interp(u,self.wind_speed,self.Ct)


def get_WAV_pp(U_i,P_i,turb,f):
    #use power production to weight-average function f
    #(there may be better ways)
    WAV = np.sum(f(U_i)*turb.Cp_f(U_i)*P_i*U_i**3/np.sum(turb.Cp_f(U_i)*P_i*U_i**3))
    return WAV

class iea_10_for_tests():

    def __init__(self, turbine):

        self.wind_speed = np.array([0.0, 2.9, 3.0, 4.0, 4.5147, 5.0008, 5.4574, 5.8833, 6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484, 7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337, 8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717, 9.3202, 9.6035, 9.921, 10.272, 10.6557, 10.7577, 11.5177, 11.9941, 12.4994, 13.0324, 13.592, 14.1769, 14.7859, 15.4175, 16.0704, 16.7432, 17.4342, 18.1421, 18.8652, 19.6019, 20.3506, 21.1096, 21.8773, 22.6519, 23.4317, 24.215, 25.01, 25.02, 50.0])

        self.Ct = turbine.ct(self.wind_speed)

        ideal_power = 0.5 * 1.225 * 99**2 * np.pi * self.wind_speed**3
        self.Cp = turbine.power(self.wind_speed) / ideal_power

        self.Z_h = 119.0 #height
        self.D = 198.0 #diameter
        self.A = np.pi*(self.D/2)**2 #area
        self.U = 25. #cut out speed
        self.name = 'iea_10mw'

    def Cp_f(self,u):
        return np.interp(u,self.wind_speed,self.Cp)

    def Ct_f(self,u):
        return np.interp(u,self.wind_speed,self.Ct)







