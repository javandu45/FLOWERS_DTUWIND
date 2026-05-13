from FLOWERS_original import *
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from utils import *
from scipy.special import gamma
import numpy as np

from FLOWERS_integrated import NOJ_flowers, gaussian_flowers

import warnings
warnings.filterwarnings("ignore")

def noj_flowers_from_paper(site, x, y):

    """
    NOJ FLOWERS AEP uses NREL- 5 MW as default turbine
    """
    
    freqs = site.ds.Sector_frequency.values[:-1]
    avg_ws = site.ds.Weibull_A.values * gamma(1 + 1/site.ds.Weibull_k.values)
    avg_ws = avg_ws[:-1]
    freqs = freqs / np.sum(freqs)

    wind_rose = pd.DataFrame({"wd": np.arange(0, 360, 1), "ws": avg_ws, "freq_val": freqs})

    flowers_model = FlowersInterface(wind_rose=wind_rose, layout_x=x, layout_y=y, num_terms=20)

    return flowers_model.calculate_aep()/1e9


def gaussian_flowers_from_paper(site, windTurbine, x, y):

    """
    Gaussian FLOWERS AEP uses IEA 10 MW as default turbine
    """

    turb = iea_10_for_tests(windTurbine)
    P_i = site.ds.Sector_frequency.values[:-1]
    U_i = site.ds.Weibull_A.values * gamma(1 + 1/site.ds.Weibull_k.values)
    U_i = U_i[:-1]
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

site = generic_site(10)
x, y = Hornsrev1Site().initial_position.T

######## NO JENSEN ########

nrel_5MW_turbine = nrel_5MW()
x, y = generate_array(200, turbine=nrel_5MW_turbine)

print("--- AEP COMPARISON - 200 5MW turbines ---")

# Jessen Flowers from paper
aep = noj_flowers_from_paper(site=site, x=x, y=y)
print(f'AEP for NOJ FLOWERS from paper: {aep:.2f} GWh')

# Jensen Flowers from new code
wfm = NOJ_flowers(site=site, WindTurbine=nrel_5MW_turbine, n_terms=20, k=0.05)
aep = wfm.aep(x=x, y=y)
print(f'AEP for NOJ FLOWERS from new code: {aep:.4f} GWh')

######## Gaussian ########

iea_10MW_turbine = IEA_10MW()
x, y = generate_array(200, turbine=iea_10MW_turbine)

print("\n--- AEP COMPARISON - 200 10MW turbines ---")

# Gaussian Flowers from paper
aep = gaussian_flowers_from_paper(site=site, windTurbine=iea_10MW_turbine, x=x, y=y)
print(f'AEP for Gaussian FLOWERS from paper: {aep:.4f} GWh')

# Gaussian Flowers from new code
wfm = gaussian_flowers(site=site, WindTurbine=iea_10MW_turbine, n_terms=20, k=0.03)
aep = wfm.aep(x=x, y=y)
print(f'AEP for Gaussian FLOWERS from new code: {aep:.4f} GWh')