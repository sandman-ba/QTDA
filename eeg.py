import numpy as np
import pandas as pd
import concurrent.futures
from itertools import product, repeat
from classicTakens import *
from persistenceDiagram import *
from persistentDirac import diracMaximalTimeSeries


##################
# Set Parameters #
##################
tau = 1 # Delay
ks = [0, 1] # Dimension for Betti number
N = 16 # Number of scales
eps0 = 0 # Smallest scale
epsStep = 1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
data = data.iloc[110:150,:]
data['time'] = data.reset_index().index
data = np.array(data.Channel2)


#######################
# Persistence Diagram #
#######################
bettis = []
for k in ks:
    dirac = diracMaximalTimeSeries(data, k, tau)

    def betti(eps):
        return persistentBetti(data, k, eps, dirac, tau)

    bettik = []

    for eps in reversed(scales):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch = executor.map(betti, product(scales, [eps]))

        bettik.append(list(batch))

    bettis.append(np.array(bettik, np.half))

persistenceDiagram(bettis, scales, figure_path='figures/diagram-eeg.png')
