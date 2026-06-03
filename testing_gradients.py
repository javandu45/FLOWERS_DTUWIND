from FLOWERS import TurbOPark_flowers

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
import warnings
warnings.filterwarnings("ignore")
import time

# Coordinates for testing
x, y = Hornsrev1Site().initial_position.T

flowers_model_tp = TurbOPark_flowers(site=Hornsrev1Site(),
                                    windTurbines=V80(),
                                    n_terms=6,
                                    ti=0.1)

aep = flowers_model_tp.aep(x=x, y=y)

grad_x_ag, grad_y_ag = flowers_model_tp.aep_gradient(gradient_method="Autograd", x=x, y=y)
grad_x_exp, grad_y_exp = flowers_model_tp.aep_gradient(gradient_method="Exact", x=x, y=y)
