import numpy as np
from itertools import product, repeat
import concurrent.futures
from classicTakens import persistentBetti
from persistenceDiagram import *


###############
# Two Squares #
###############
k = 1 # Dimension for Betti number
tau = None
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales
s = np.sqrt(2)
scales = [eps0 + (x * epsStep) for x in range(N)]
data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [10.0, 0.0], [10.0 + s, 0.0], [10.0 + s, s], [10.0, s]])


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


persistenceDiagram(bettis, scales, output_path='results/squares/', figure_path='figures/diagram-squares.png', save_data=True, save_figure=True)
