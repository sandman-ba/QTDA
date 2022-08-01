import numpy as np
from itertools import product, repeat
import concurrent.futures
from classicTakens import persistentBetti
from persistenceDiagram import *


#################
# One Attractor #
#################
T = 5 # Number of points
k = 1 # Dimension for Betti number
tau = 1 # Delay
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series



#######################
# Persistence Diagram #
#######################
with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(persistentBetti, repeat(data), repeat(k), product(scales, reversed(scales)), repeat(tau))


persistenceDiagram(bettis, scales, output_path='results/one-period/', figure_path='figures/diagram-one-period.png', save_data=True, save_figure=True)



