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
tau = 8 # Delay
ks = [0, 1] # Dimension for Betti number
N = 16 # Number of scales
eps0 = 0 # Smallest scale
epsStep = 1 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2', 'Channel1'])
data = data.iloc[5857:,:]
data = data.reset_index()
data = data.iloc[1110:1160,:]
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

