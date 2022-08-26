import numpy as np
from numpy import pi
from itertools import product, repeat
import concurrent.futures
from classicTakens import *
from persistenceDiagram import *
from persistentDirac import diracMaximalTimeSeries

##################
# Set Parameters #
##################
T = 12
tau = 2 # Delay
ks = [0, 1] # Dimension for Betti number
N = 25 # Number of scales
eps0 = 0 # Smallest scale
epsStep = 0.1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series


#######################
# Persistence Diagram #
#######################
bettis = []
for k in ks:
    dirac = diracMaximalTimeSeries(data, k, tau)

    def betti(eps):
        return persistentBetti(data, k, eps, dirac, tau)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        bettik = executor.map(betti, product(scales, reversed(scales)))

    bettis.append(np.fromiter(bettik, np.double))

persistenceDiagram(bettis, scales, figure_path='figures/diagram-two-period.png')
