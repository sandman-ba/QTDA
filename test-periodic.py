import numpy as np
from numpy import pi
from itertools import product, repeat
import concurrent.futures
from classicTakens import *
from persistenceDiagram import *
from persistentDirac import diracMaximalTimeSeries


#################
# One Attractor #
#################
T = 5 # Number of points
ks = [0] # Dimension for Betti number
tau = 1 # Delay
N = 25 # Number of scales
eps0 = 0 # Smallest scale
epsStep = 0.1 # Step between scales
scales = [round(10 * (eps0 + (x * epsStep))) / 10 for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series



#######################
# Persistence Diagram #
#######################
bettis = []
for k in ks:
    dirac = diracMaximalTimeSeries(data, k, tau)

    def betti(eps):
        return persistentBettiClassic(data, k, eps, dirac, tau)

    bettik = []

    for eps in reversed(scales):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch = executor.map(betti, product(scales, [eps]))

        bettik.append(list(batch))

    bettis.append(np.array(bettik, np.half))

persistenceDiagram(bettis, scales, figure_path='figures/diagram-one-period-test.png')
