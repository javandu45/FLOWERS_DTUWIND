from FLOWERS import TurbOPark_flowers, gaussian_flowers, NOJ_flowers

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
import warnings
warnings.filterwarnings("ignore")
import time
from examples.utils import generic_site

# site = generic_site(ws=10)

# Coordinates for testing
x, y = Hornsrev1Site().initial_position.T
flowers_model_g = gaussian_flowers(site=Hornsrev1Site(),
                                    windTurbines=V80(),
                                    n_terms=7)

aep = flowers_model_g.aep(x=x, y=y)

time_i = time.time()
grad_x_ag, grad_y_ag = flowers_model_g.aep_gradient(gradient_method="Autograd", x=x, y=y)
time_ag = time.time() - time_i
print(f"Autograd gradient computed in {time_ag:.10f} seconds")
print(f"Autograd gradient: grad_x_ag={grad_x_ag[:5]}")

time_i = time.time()
grad_x_exp, grad_y_exp = flowers_model_g.aep_gradient(gradient_method="Exact", x=x, y=y)
time_exp = time.time() - time_i
print(f"Exact gradient computed in {time_exp:.10f} seconds")
print(f"Exact gradient: grad_x_exp={grad_x_exp[:5]}")