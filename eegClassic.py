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
tau = 6 # Delay
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
bettisClassic = []
for k in ks:
    dirac = diracMaximalTimeSeries(data, k, tau)

    def bettiClassic(eps):
        return persistentBettiClassic(data, k, eps, dirac, tau)

    bettikClassic = []

    for eps in reversed(scales):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            batchClassic = executor.map(bettiClassic, product(scales, [eps]))

        bettikClassic.append(list(batchClassic))
    bettisClassic.append(np.array(bettikClassic, np.half))

persistenceDiagram(bettisClassic, scales, figure_path='figures/diagram-eeg-classic.png')

