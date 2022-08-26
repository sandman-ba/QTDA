import numpy as np
from itertools import product, repeat
import concurrent.futures
from classicTakens import *
from persistenceDiagram import *
from persistentDirac import diracMaximalPointCloud


###############
# Two Squares #
###############
ks = [0, 1] # Dimension for Betti number
N = 25 # Number of scales
eps0 = 0 # Smallest scale
epsStep = 0.1 # Step between scales
s = np.sqrt(2)
scales = [eps0 + (x * epsStep) for x in range(N)]
data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [10.0, 0.0], [10.0 + s, 0.0], [10.0 + s, s], [10.0, s]])



#######################
# Persistence Diagram #
#######################
bettis = []
for k in ks:
    dirac = diracMaximalPointCloud(data, k)

    def betti(eps):
        return persistentBetti(data, k, eps, dirac)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        bettik = executor.map(betti, product(scales, reversed(scales)))

    bettis.append(np.fromiter(bettik, np.double))

persistenceDiagram(bettis, scales, figure_path='figures/diagram-squares.png')
