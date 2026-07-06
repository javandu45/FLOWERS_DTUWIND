from FLOWERS import NOJ_flowers, gaussian_flowers, TurbOPark_flowers, fuga_flowers

import py_wake
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.fuga import Ott_Nielsen_2014
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.literature.turbopark import Nygaard_2022
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from py_wake.utils.gradients import autograd
import time
import matplotlib.pyplot as plt
import os
from utils import generic_site
import numpy as np

# Ignore numerical errors coming from FLOWERS (division by zero, etc.)
import warnings
warnings.filterwarnings("ignore")

# Wind farm conditions
site = Hornsrev1Site()
turbine = V80()
x, y = site.initial_position.T
# x, y = x[:20], y[:20]
site=generic_site(10)

lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'

# FLOWERS AEP models
wfm_NOJ_flowers = NOJ_flowers(site=site, windTurbines=turbine, n_terms=6)
wfm_gaussian_flowers = gaussian_flowers(site=site, windTurbines=turbine, n_terms=6)
wfm_turbopark_flowers = TurbOPark_flowers(site=site, windTurbines=turbine, n_terms=6, ti=0.1)
wfm_fuga_flowers = fuga_flowers(site=site, windTurbines=turbine, n_terms=10, dx=None)

# Convential AEP models
wfm_NOJ = Jensen_1983(site=site, windTurbines=turbine, k=0.04)
wfm_gaussian = Bastankhah_PorteAgel_2014(site=site, windTurbines=turbine, k=0.03)
wfm_fuga = Ott_Nielsen_2014(site=site, windTurbines=turbine, LUT_path=lut_path)

wfm_eval = wfm_fuga

##########################################################
# Initial AEP computation
sim_res = wfm_eval(x, y,
            wd=site.default_wd,
            ws=site.default_ws)

initial_aep = sim_res.aep().sum().values

##########################################################
# Optimization problem setup - Using Fuga FLOWERS
wfm = wfm_fuga_flowers

# AEP function
def aep_func(x, y):
    return wfm.aep(x, y)

# AEP gradients
def aep_gradient(x, y):
    return wfm.aep_gradient(gradient_method="Autograd", wrt_arg=["x","y"], x=x, y=y)

# Setup constraints
min_dist = 5*turbine.diameter()
turb_sep_constraint = SpacingConstraint(min_dist)
boundary = [[x.min(), y.min()], [x.max(), y.min()], [x.max(), y.max()], [x.min(), y.max()], [x.min(), y.min()]]
boundary_constraint = XYBoundaryConstraint(boundary)

constraints = [turb_sep_constraint, boundary_constraint]

# Setup cost component
aep_comp = CostModelComponent(input_keys=[('x', x),('y', y)],
                              n_wt=len(x),
                              cost_function=aep_func,
                              cost_gradient_function=aep_gradient,
                              maximize=True,
                              objective=True,
                              output_keys=['AEP'])

# Set up driver
driver = EasyScipyOptimizeDriver(optimizer="SLSQP",
                                 maxiter=50,
                                 tol=10)

# Set up optimization problem
topfarm_problem = TopFarmProblem(design_vars=dict(zip(['x', 'y'], [x, y])),
                                 cost_comp=aep_comp,
                                 driver=driver,
                                 constraints=constraints,
                                 n_wt=len(x),
                                 expected_cost=1e-5)


# Running optimization
time_start = time.time()
print("Starting optimization...")
_, state, recorder = topfarm_problem.optimize()
time_end = time.time()
time_fuga = time_end - time_start

x_opt = state["x"]
y_opt = state["y"]

# Evaluate final layout using PyWake
sim_res = wfm_eval(x_opt, y_opt,
            wd=site.default_wd,
            ws=site.default_ws)

final_aep_fuga = sim_res.aep().sum().values


##########################################################
# Optimization problem setup - Using NOJ FLOWERS
wfm = wfm_NOJ_flowers

# AEP function
def aep_func(x, y):
    return wfm.aep(x, y)

# AEP gradients
def aep_gradient(x, y):
    return wfm.aep_gradient(gradient_method="Autograd", wrt_arg=["x","y"], x=x, y=y)

# Setup cost component
aep_comp = CostModelComponent(input_keys=[('x', x),('y', y)],
                              n_wt=len(x),
                              cost_function=aep_func,
                              cost_gradient_function=aep_gradient,
                              maximize=True,
                              objective=True,
                              output_keys=['AEP'])

# Set up optimization problem
topfarm_problem = TopFarmProblem(design_vars=dict(zip(['x', 'y'], [x, y])),
                                 cost_comp=aep_comp,
                                 driver=driver,
                                 constraints=constraints,
                                 n_wt=len(x),
                                 expected_cost=1e-5)


# Running optimization
time_start = time.time()
print("Starting optimization...")
_, state, recorder = topfarm_problem.optimize()
time_end = time.time()
time_NOJ = time_end - time_start

x_opt = state["x"]
y_opt = state["y"]

# Evaluate final layout using PyWake
sim_res = wfm_eval(x_opt, y_opt,
            wd=site.default_wd,
            ws=site.default_ws)

final_aep_NOJ = sim_res.aep().sum().values

##########################################################
# Optimization problem setup - Using Fuga FLOWERS
wfm = wfm_fuga

# AEP function
def aep_func(x, y):
    return wfm.aep(x, y)

# AEP gradients
def aep_gradient(x, y):

    jx, jy = wfm.aep_gradients(gradient_method=autograd,
                                            x=x, 
                                            y=y,
                                            ws=site.default_ws, 
                                            wd=site.default_wd)
                                            
    daep = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])
    return daep

# Setup cost component
aep_comp = CostModelComponent(input_keys=[('x', x),('y', y)],
                              n_wt=len(x),
                              cost_function=aep_func,
                              cost_gradient_function=aep_gradient,
                              maximize=True,
                              objective=True,
                              output_keys=['AEP'])

# Set up optimization problem
topfarm_problem = TopFarmProblem(design_vars=dict(zip(['x', 'y'], [x, y])),
                                 cost_comp=aep_comp,
                                 driver=driver,
                                 constraints=constraints,
                                 n_wt=len(x),
                                 expected_cost=1e-5)


# Running optimization
time_start = time.time()
print("Starting optimization...")
_, state, recorder = topfarm_problem.optimize()
time_end = time.time()
time_fuga_pw = time_end - time_start

x_opt = state["x"]
y_opt = state["y"]

# Evaluate final layout using PyWake
sim_res = wfm_eval(x_opt, y_opt,
            wd=site.default_wd,
            ws=site.default_ws)

final_aep_fuga_pw = sim_res.aep().sum().values


print()
print(f"Initial AEP from flow model: {initial_aep:.2f} GWh")
print("--------------------------------------------------")
print(f"Optimization with Fuga FLOWERS completed in {time_fuga:.2f} seconds")
print(f"Final AEP from pywake fuga wind farm model: {final_aep_fuga:.2f}")
print("--------------------------------------------------")
print(f"Optimization with NOJ FLOWERS completed in {time_NOJ:.2f} seconds")
print(f"Final AEP from pywake noj wind farm model: {final_aep_NOJ:.2f}")
print("--------------------------------------------------")
print(f"Optimization with Fuga PyWake completed in {time_fuga_pw:.2f} seconds")
print(f"Final AEP from pywake fuga wind farm model: {final_aep_fuga_pw:.2f}")