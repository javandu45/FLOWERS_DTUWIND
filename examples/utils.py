from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
from py_wake.site import UniformWeibullSite
import pandas as pd
import numpy as np
from pathlib import Path

# The following functions are for demonstration purposes only, and not part of the package

def generic_site(ws=11):

    """
    Generate a UniformWeibullSite based on wind speed characteristics from a CSV file.
    """

    wind_char = pd.read_csv(Path(__file__).resolve().parent / f"wind_rose_{ws}.csv", index_col=0)

    wind_char = wind_char.reset_index()

    site = UniformWeibullSite(
        p_wd = wind_char["P"],
        a = wind_char["A"],
        k = wind_char["k"],
        ti = 0.1    
    )

    return site


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


def IEA_10MW():

    """
    Create a WindTurbine object representing the IEA 10MW turbine using tabular Cp and Ct data.
    """

    diameter = 198.0

    cp_table = np.array([0.0, 0.0, 0.074, 0.3251, 0.3762, 0.4027, 0.4156, 0.423, 0.4274, 0.4293, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4298, 0.4305, 0.438256, 0.425908, 0.347037, 0.307306, 0.271523, 0.239552, 0.211166, 0.186093, 0.164033, 0.144688, 0.12776, 0.112969, 0.100062, 0.0888, 0.078975, 0.070401, 0.062913, 0.056368, 0.05064, 0.04562, 0.041216, 0.037344, 0.033935, 0.0, 0.0])

    ct_table = np.array([0.0, 0.0, 0.7701, 0.7701, 0.7763, 0.7824, 0.782, 0.7802, 0.7772, 0.7719, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7675, 0.7651, 0.7587, 0.5056, 0.431, 0.3708, 0.3209, 0.2788, 0.2432, 0.2128, 0.1868, 0.1645, 0.1454, 0.1289, 0.1147, 0.1024, 0.0918, 0.0825, 0.0745, 0.0675, 0.0613, 0.0559, 0.0512, 0.047, 0.0, 0.0])

    u_table =  np.array([0.0, 2.9, 3.0, 4.0, 4.5147, 5.0008, 5.4574, 5.8833, 6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484, 7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337, 8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717, 9.3202, 9.6035, 9.921, 10.272, 10.6557, 10.7577, 11.5177, 11.9941, 12.4994, 13.0324, 13.592, 14.1769, 14.7859, 15.4175, 16.0704, 16.7432, 17.4342, 18.1421, 18.8652, 19.6019, 20.3506, 21.1096, 21.8773, 22.6519, 23.4317, 24.215, 25.01, 25.02, 50.0])

    p_table = 0.5 * 1.225 * (198.0**2 / 4) * np.pi * u_table**3 * cp_table

    wt = WindTurbine(name="IEA_10MW",
                    diameter=diameter,
                    hub_height=119,
                    powerCtFunction=PowerCtTabular(ws=u_table, power=p_table, power_unit="W", ct=ct_table, method="linear"))
    
    return wt


def generate_array(n_tur, turbine, spacing=5, limits="Square"):

    """
    Generate a grid array of wind turbines based on the number of turbines, turbine diameter, and spacing factor.
    """
    
    if limits == "Square":
        D = turbine.diameter()
        n_gaps = int(np.ceil(np.sqrt(n_tur)))  # Ensure enough gaps for 100 turbines
        x = np.arange(0, n_gaps * D * spacing, D * spacing)
        y = np.arange(0, n_gaps * D * spacing, D * spacing)
        x, y = np.meshgrid(x, y)
        x = x.flatten()[:n_tur] 
        y = y.flatten()[:n_tur]

        return x.flatten(), y.flatten()
