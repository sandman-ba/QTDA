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
k = 1 # Dimension for Betti number
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series


#######################
# Persistence Diagram #
#######################
dirac = diracMaximalTimeSeries(data, k, tau)

def bettiClassic(eps):
    return persistentBettiClassic(data, k, eps, dirac, tau)

def betti(eps):
    return persistentBetti(data, k, eps, dirac, tau, M_multiplier=10)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettisClassic = executor.map(bettiClassic, product(scales, reversed(scales)))

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, product(scales, reversed(scales)))


persistenceDiagram(bettisClassic, scales, figure_path='figures/diagram-two-period-classic.png', save_figure=True)

persistenceDiagram(bettis, scales, output_path='results/two-period/', figure_path='figures/diagram-two-period.png', save_data=True, save_figure=True)




