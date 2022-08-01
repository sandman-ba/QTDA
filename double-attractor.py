import numpy as np
from numpy import pi
from itertools import product, repeat
import concurrent.futures
from classicTakens import persistentBetti
from persistenceDiagram import *

##################
# Set Parameters #
##################
T = 12
tau = 2 # Delay
k = 1 # Dimension for Betti number
N = 20 # Number of scales
eps0 = 0.2 # Smallest scale
epsStep = 0.1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series

#######################
# Persistence Diagram #
#######################
with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(persistentBetti, repeat(data), repeat(k), product(scales, reversed(scales)), repeat(tau))


persistenceDiagram(bettis, scales, output_path='results/two-period/', figure_path='figures/diagram-two-period.png', save_data=True, save_figure=True)



