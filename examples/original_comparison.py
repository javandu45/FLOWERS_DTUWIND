from FLOWERS.noj import NOJ_flowers
from FLOWERS.bastankhah import gaussian_flowers
from utils import *
from scipy.special import gamma
import numpy as np
from FLOWERS.original import *

import warnings
warnings.filterwarnings("ignore")

def noj_flowers_from_paper(site, x, y):

    """
    Compute the AEP using NOJ FLOWERS with the paper's original code
    NOJ FLOWERS AEP uses NREL- 5 MW as default turbine
    """
    
    freqs = site.ds.Sector_frequency.values
    avg_ws = site.ds.Weibull_A.values * gamma(1 + 1/site.ds.Weibull_k.values)
    avg_ws = avg_ws
    freqs = freqs / np.sum(freqs)

    wind_rose = pd.DataFrame({"wd": np.arange(0, 361, 1), "ws": avg_ws, "freq_val": freqs})
    wind_rose = {"wd": np.arange(0, 361, 1), "ws": avg_ws, "freq_val": freqs}

    flowers_model = FlowersInterface(wind_rose=wind_rose, layout_x=x, layout_y=y, num_terms=20, turbine="nrel_5MW")

    return flowers_model.calculate_aep()/1e9


def gaussian_flowers_from_paper(site, x, y):

    """
    Compute the AEP using Gaussian FLOWERS with the paper's original code
    Gaussian FLOWERS AEP uses IEA 10 MW as default turbine
    """

    turb = iea_10MW()
    P_i = site.ds.Sector_frequency.values
    P_i = P_i / np.sum(P_i)
    U_i = site.ds.Weibull_A.values * gamma(1 + 1/site.ds.Weibull_k.values)
    U_i = U_i
    _,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
    wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
    layout = np.array([x, y]).T

    aep_func_d = lambda: ntag_PA(Fourier_coeffs3_PA,
                                        layout,
                                        layout,
                                        turb,
                                        0.03, 
                                        #(Ct_op = 3 cnst) 
                                        #(Cp_op = 2 global )    
                                        wav_Ct)

    (powj_d,_),time_2 = adaptive_timeit(aep_func_d,timed=False)
    aep2 = np.sum(powj_d)

    return aep2


# Obtain pywake site based on a wind rose from CSV file (avg ws of around 10 m/s)
site = generic_site(10)

############ NO JENSEN ############

nrel_5MW_turbine = nrel_5MW()

# Generate regular layout of 200 turbines with 5D separation
x, y = generate_array(200, turbine=nrel_5MW_turbine)

print("--- AEP COMPARISON - 200 5MW turbines ---")

# Jessen Flowers from paper
aep = noj_flowers_from_paper(site=site, x=x, y=y)
print(f'AEP for NOJ FLOWERS from paper: {aep:.4f} GWh')

# Jensen Flowers from new code
wfm = NOJ_flowers(site=site, windTurbines=nrel_5MW_turbine, n_terms=20, k=0.05)
aep = wfm.aep(x=x, y=y)
print(f'AEP for NOJ FLOWERS from new code: {aep:.4f} GWh')

############ Gaussian ############

iea_10MW_turbine = IEA_10MW()
# Generate regular layout of 200 turbines with 5D separation
x, y = generate_array(200, turbine=iea_10MW_turbine)

print("\n--- AEP COMPARISON - 200 10MW turbines ---")

# Gaussian Flowers from paper
aep = gaussian_flowers_from_paper(site=site, x=x, y=y)
print(f'AEP for Gaussian FLOWERS from paper: {aep:.4f} GWh')

# Gaussian Flowers from new code
wfm = gaussian_flowers(site=site, windTurbines=iea_10MW_turbine, n_terms=20, k=0.03)
aep = wfm.aep(x=x, y=y)
print(f'AEP for Gaussian FLOWERS from new code: {aep:.4f} GWh')