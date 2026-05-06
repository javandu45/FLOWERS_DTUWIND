from FLOWERS import *
import numpy as np

from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80

site = Hornsrev1Site()

flowers_aep = flowers(WindTurbine= V80(),
                      site=Hornsrev1Site(),
                      n_terms=4)

x = [0, 100, 200]
y = [0, 0, 0]

coords = (x, y)

AEP = flowers_aep.calculate_AEP(x=x, y=y)

print("AEP:", AEP)

AEP_grad_x, AEP_grad_y = flowers_aep.calculate_gradients(coords=coords, method="Exact")

print("AEP grad X:", AEP_grad_x)
print("AEP grad Y:", AEP_grad_y)