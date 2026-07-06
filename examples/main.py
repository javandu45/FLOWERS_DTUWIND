# %%
# Demonstration of FLOWERS AEP usage

# Imports
from FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers, fuga_flowers
from utils import generic_site

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.noj import Jensen_1983
from py_wake.utils.gradients import autograd as pw_autograd
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.fuga import Ott_Nielsen_2014
import py_wake

import time
import numpy as np
import os

# Ignore numerical errors coming from FLOWERS (division by zero, etc.)
import warnings
warnings.filterwarnings("ignore")

# Coordinates for testing
x, y = Hornsrev1Site().initial_position.T
lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'
# site = generic_site(10)
site = Hornsrev1Site()

# %%
# ##################################################
# AEP computation
# ##################################################

# ---------------------------
# USING FLOWERS
# ---------------------------

# Initializing FLOWERS model
# In this case, given that Horsrev1Site has a wind rose with only 12 wind directions,
# we only use 6 fourier terms (max number of Fourier Modes = n_wd/2)

# NO Jensen Flowers
flowers_model_noj = NOJ_flowers(site=site,
                        windTurbines=V80(),
                        n_terms=6)

# Gaussian Bastankhah Flowers
flowers_model_g = gaussian_flowers(site=site,
                                   windTurbines=V80(),
                                   n_terms=6)

# TurboPark Flowers
flowers_model_tp = TurbOPark_flowers(site=site,
                                    windTurbines=V80(),
                                    n_terms=6,
                                    ti=0.1)

# Fuga Flowers
flowers_model_fuga = fuga_flowers(site=site,
                                  windTurbines=V80(),
                                  n_terms=6,
                                  dx=None)


# AEP computation
time_i = time.time()
AEP = flowers_model_noj.aep(x=x, y=y)
total_time = time.time() - time_i

print("### AEP COMPUTATION COMPARISON ###")
print("--- FLOWERS ---")
print(f"\t NOJ AEP: {AEP:.2f} GWh")
print(f"\t NOJ AEP time: {total_time:.5f} s")

time_i = time.time()
AEP = flowers_model_g.aep(x=x, y=y)
total_time = time.time() - time_i

print(f"\t Gaussian AEP: {AEP:.2f} GWh")
print(f"\t Gaussian AEP time: {total_time:.5f} s")

time_i = time.time()
AEP = flowers_model_tp.aep(x=x, y=y)
total_time = time.time() - time_i

print(f"\t TurboPark AEP: {AEP:.2f} GWh")
print(f"\t TurboPark AEP time: {total_time:.5f} s")

time_i = time.time()
AEP = flowers_model_fuga.aep(x=x, y=y)
total_time = time.time() - time_i

print(f"\t Fuga AEP: {AEP:.2f} GWh")
print(f"\t Fuga AEP time: {total_time:.5f} s")

# ---------------------------
# USING CONVENTIONAL METHODS IN PYWAKE
# ---------------------------

# Initializing models
# NO Jensen
flow_model_noj = Jensen_1983(site = site,
                        windTurbines = V80(),
                        k=0.04)

# Gaussian Bastankhah
flow_model_g = Bastankhah_PorteAgel_2014(site = site,
                                    windTurbines = V80(),
                                    k = 0.03)

# TurboPark
flow_model_tp = Nygaard_2022(site = site,
                            windTurbines = V80())

# Fuga
flow_model_fuga = Ott_Nielsen_2014(LUT_path=lut_path, 
                                   site = site,
                                   windTurbines = V80())

# AEP computation
time_i = time.time()
sim_res = flow_model_noj(x, y,
                     wd=site.default_wd,
                     ws=site.default_ws)
AEP = sim_res.aep().sum().values
total_time = time.time() - time_i

print("--- CONVENTIONAL ---")
print(f"\t NOJ AEP: {AEP:.2f} GWh")
print(f"\t NOJ AEP time: {total_time:.5f} s")

time_i = time.time()
sim_res = flow_model_g(x, y,
                     wd=site.default_wd,
                     ws=site.default_ws)
AEP = sim_res.aep().sum().values
total_time = time.time() - time_i

print(f"\t Gaussian AEP: {AEP:.2f} GWh")
print(f"\t Gausssian AEP time: {total_time:.5f} s")

# time_i = time.time()
# sim_res = flow_model_tp(x, y,
#                      wd=site.default_wd,
#                      ws=site.default_ws)
# AEP = sim_res.aep().sum().values
# total_time = time.time() - time_i

# print(f"\t TurboPark AEP: {AEP:.2f} GWh")
# print(f"\t TurboPark AEP time: {total_time:.5f} s")

time_i = time.time()
sim_res = flow_model_fuga(x, y,
                        wd=site.default_wd,
                        ws=site.default_ws)
AEP = sim_res.aep().sum().values
total_time = time.time() - time_i

print(f"\t Fuga AEP: {AEP:.2f} GWh")
print(f"\t Fuga AEP time: {total_time:.5f} s")

# %%
# ##################################################
# Gradient computation
# ##################################################

# ---------------------------
# USING FLOWERS
# ---------------------------

coords = (x, y)

# Analytical gradient - NOJ
time_i = time.time()
AEP = flowers_model_noj.aep_gradient(gradient_method="Exact", wrt_arg=["x"], x=x, y=y)
total_time = time.time() - time_i

print("--------------------------------------------------")
print("### AEP GRADIENTS COMPARISON ###")
print("--- FLOWERS ---")
print(f"\t NOJ analytical differentiation time: {total_time:.5f} s")

# Automatic differentiation gradients - Gaussian
time_i = time.time()
AEP = flowers_model_g.aep_gradient(gradient_method="Exact", wrt_arg=["x", "y"], x=x, y=y)
total_time = time.time() - time_i

print(f"\t Gaussian analytical differentiation time: {total_time:.5f} s")

# Automatic differentiation gradients - TurbOPark
time_i = time.time()
AEP = flowers_model_tp.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)
total_time = time.time() - time_i

print(f"\t TurboPark analytical differentiation time: {total_time:.5f} s")

# Automatic differentiation gradients - Fuga
time_i = time.time()
AEP = flowers_model_fuga.aep_gradient(gradient_method="Autograd", wrt_arg=["x", "y"], x=x, y=y)
total_time = time.time() - time_i

print(f"\t Fuga analytical differentiation time: {total_time:.5f} s")

# ---------------------------
# USING CONVENTIONAL METHODS IN PYWAKE
# ---------------------------

time_i = time.time()
jx, jy = flow_model_noj.aep_gradients(gradient_method=pw_autograd,
                                        wrt_arg=['x', 'y'],
                                        x=x, 
                                        y=y,
                                        wd=site.default_wd)
total_time = time.time() - time_i

print("--- CONVENTIONAL ---")
print(f"\t NOJ Gradients: {total_time:.5f} s")

time_i = time.time()
jx, jy = flow_model_g.aep_gradients(gradient_method=pw_autograd,
                                        wrt_arg=['x', 'y'],
                                        x=x, 
                                        y=y,
                                        wd=site.default_wd)
total_time = time.time() - time_i

print(f"\t Gaussian Gradients: {total_time:.5f} s")


time_i = time.time()
jx, jy = flow_model_tp.aep_gradients(gradient_method=pw_autograd,
                                        wrt_arg=['x', 'y'],
                                        x=x, 
                                        y=y,
                                        wd=site.default_wd)
total_time = time.time() - time_i


print(f"\t TurboPark Gradients: {total_time:.5f} s")

time_i = time.time()
jx, jy = flow_model_fuga.aep_gradients(gradient_method=pw_autograd,
                                        wrt_arg=['x', 'y'],
                                        x=x, 
                                        y=y,
                                        wd=site.default_wd)

total_time = time.time() - time_i

print(f"\t Fuga Gradients: {total_time:.5f} s")


# %%
####################################################
# AEP per turbine visualization
####################################################

flowers_model_noj.plot_AEP_per_turbine(x, y)

# The visualization for Gaussian makes sense, althought the values do not
# make much sense. They are all extremely similar
flowers_model_g.plot_AEP_per_turbine(x, y)

flowers_model_tp.plot_AEP_per_turbine(x, y)

flowers_model_fuga.plot_AEP_per_turbine(x, y)

# %%

r0 = -5
alpha = np.linspace(-np.pi, np.pi, 360, endpoint=False)
g_true = flowers_model_fuga._LUT(np.broadcast_to(r0, alpha.shape), alpha)   # raw, sign as used
A, B = flowers_model_fuga._evaluate_moments(np.array([[r0]]))
A = A[0,0]; B = B[0,0]
m = np.arange(len(A))
g_recon = A[0]/2 + (A[1:]*np.cos(m[1:]*alpha[:,None]) + B[1:]*np.sin(m[1:]*alpha[:,None])).sum(1)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(alpha, g_true, label='true')
plt.plot(alpha, g_recon, label='reconstructed')
plt.legend()
plt.show()
