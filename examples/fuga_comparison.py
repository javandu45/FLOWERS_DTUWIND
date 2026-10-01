# %%
from FLOWERS.fuga import fuga_flowers

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.utils.gradients import autograd
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from py_wake.literature.fuga import Ott_Nielsen_2014
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver

import time
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

site = Hornsrev1Site()
turbine = V80()
x, y = site.initial_position.T

# LUT paths for LUT and FLUT
data_directory = Path(__file__).parent.parent / "data"
FLUT_PATH = str(data_directory / "fLUTs_Zeta0=0.00e+00_8_16_D80_zhub70_zi400_z0=0.00010000_z70.0_UL.nc")
LUT_PATH = str(data_directory / "LUTs_Zeta0=0.00e+00_8_16_D80_zhub70_zi400_z0=0.00010000_z70.0_UL_nx2048_ny512_dx20_dy5.NC")

# %% ----- Initialization of FLOWERS AEP models -----
# 1: Using linear summation derivation approach, LUTS
# 2: Using linear summation derivation approach, FLUTS
# 3: Using binomial expansion derivation approach, LUTS
# 4: Using binomial expansion derivation approach, FLUTS
# 5: Using PyWake, LUTS

# 1: Fuga FLOWERS using NOJ derivation approach
time_start = time.time()
wfm_1 = fuga_flowers(site=site, windTurbines=turbine, n_terms=7, lut_file=LUT_PATH, source="lut")
time_1_initialization = time.time() - time_start

# 2: Fuga FLOWERS using NOJ derivation approach, FLUTS
time_start = time.time()
wfm_2 = fuga_flowers(site=site, windTurbines=turbine, n_terms=7, lut_file=FLUT_PATH, source="flut")
time_2_initialization = time.time() - time_start

# 3: Fuga FLOWERS using gaussian derivation approach, LUTS
time_start = time.time()
wfm_3 = fuga_flowers(site=site, windTurbines=turbine, n_terms=7, source="lut", lut_file=LUT_PATH, method="binomial")
time_3_initialization = time.time() - time_start

# 4: Fuga FLOWERS using gaussian derivation approach, FLUTS
time_start = time.time()
wfm_4 = fuga_flowers(site=site, windTurbines=turbine, n_terms=7, source="flut", lut_file=FLUT_PATH, method="binomial")
time_4_initialization = time.time() - time_start

# 5: PyWake Fuga model using LUTS
time_start = time.time()
wfm_5 = Ott_Nielsen_2014(site=site, windTurbines=turbine, LUT_path=LUT_PATH)
time_5_initialization = time.time() - time_start

# %% ----- AEP computation for each model -----
time_start = time.time()
aep_1 = wfm_1.aep(x, y)
time_1 = time.time() - time_start

time_start = time.time()
aep_2 = wfm_2.aep(x, y)
time_2 = time.time() - time_start

time_start = time.time()
aep_3 = wfm_3.aep(x, y)
time_3 = time.time() - time_start

time_start = time.time()
aep_4 = wfm_4.aep(x, y)
time_4 = time.time() - time_start

time_start = time.time()
sim_res = wfm_5(x, y,
            wd=site.default_wd,
            ws=site.default_ws)
aep_5 = sim_res.aep().sum().values
time_5 = time.time() - time_start

results = [
	("LUTS, Linear", time_1_initialization, time_1, aep_1),
	("FLUTS, Linear", time_2_initialization, time_2, aep_2),
	("LUTS, Binomial", time_3_initialization, time_3, aep_3),
	("FLUTS, Binomial", time_4_initialization, time_4, aep_4),
	("PyWake", time_5_initialization, time_5, aep_5)
]

print("--- AEP computation times ---")
print(f"{'Model':<18} {'Initialization Time [s]':>20} {'Computation Time [s]':>20} {'AEP [GWh]':>12}")
print("-" * 68)
for model, init_time, comp_time, aep in results:
	print(f"{model:<18} {init_time:>20.5f} {comp_time:>20.5f} {aep:>12.2f}")

# %% ----- WFLO problem for each model -----
def optimization_problem(wfm):

    expected_cost = 1e-4

	# AEP function
    def aep_func(x, y):
        return wfm.aep(x, y)

    # AEP gradients
    def aep_gradient(x, y):
        return wfm.aep_gradient(x=x, y=y)
    
    if i==4:
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
                                    maxiter=100,
                                    tol=10,
                                    disp=False)

    # Set up optimization problem
    topfarm_problem = TopFarmProblem(design_vars=dict(zip(['x', 'y'], [x, y])),
                                    cost_comp=aep_comp,
                                    driver=driver,
                                    constraints=constraints,
                                    n_wt=len(x),
                                    expected_cost=expected_cost)

    # Running optimization
    time_start = time.time()
    print("Starting optimization...")
    _, state, recorder = topfarm_problem.optimize()
    time_end = time.time()
    time_optimization = time_end - time_start

    x_opt = state["x"]
    y_opt = state["y"]
    # Evaluate final layout using PyWake
    sim_res = wfm_5(x_opt, y_opt,
                wd=site.default_wd,
                ws=site.default_ws)

    final_aep = sim_res.aep().sum().values

    cost = recorder.get("cost")

    # Uncomment to plot convergence plots
    plt.figure()
    plt.plot(-cost, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("AEP [GWh]")
    plt.title(f"Optimization using wfm {i + 1}")
    plt.show()

    return time_optimization, final_aep

wfms = [wfm_1, wfm_2, wfm_3, wfm_4, wfm_5]
optimization_aeps = []
optimization_times = []

for i, wfm in enumerate(wfms):
     time_optimization, final_aep = optimization_problem(wfm)
     optimization_times.append(time_optimization)
     optimization_aeps.append(final_aep)

print("--- Optimization results - Evaluated with PyWake ---")
print(f"{'Model':<18} {'Optimization Time [s]':>20} {'Final AEP [GWh]':>20}")
print("-" * 60)
for i, (model, opt_time, final_aep) in enumerate(zip(["LUTS, Linear", "FLUTS, Linear", "LUTS, Binomial", "FLUTS, Binomial", "PyWake"], optimization_times, optimization_aeps)):
    print(f"{model:<18} {opt_time:>20.5f} {final_aep:>20.2f}")