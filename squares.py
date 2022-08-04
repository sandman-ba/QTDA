import numpy as np
from itertools import product, repeat
import concurrent.futures
from classicTakens import persistentBetti
from persistenceDiagram import *


###############
# Two Squares #
###############
k = 1 # Dimension for Betti number
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales
s = np.sqrt(2)
scales = [eps0 + (x * epsStep) for x in range(N)]
data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [10.0, 0.0], [10.0 + s, 0.0], [10.0 + s, s], [10.0, s]])



#######################
# Persistence Diagram #
#######################
dirac = diracMaximalPointCloud(data, k)

def betti(eps):
    return persistentBetti(data, k, eps, dirac)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, product(scales, reversed(scales)))


persistenceDiagram(bettis, scales, output_path='results/squares/', figure_path='figures/diagram-squares.png', save_data=True, save_figure=True)
