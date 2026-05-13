from FLOWERS_integrated import *
import numpy as np

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80

site = Hornsrev1Site()

x, y = site.initial_position.T

wfm = NOJ_flowers(site=site, WindTurbine=V80(), n_terms=6)

grad_function = wfm.aep_gradient(gradient_method="Autograd", wrt_arg=["x"], x=x, y=y)