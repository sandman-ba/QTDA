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
tau = 2 # Delay
k = 1 # Dimension for Betti number
N = 30 # Number of scales
eps0 = 5 # Smallest scale
epsStep = 0.5 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
#data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2'])
#data = data.iloc[100:150,:]
data = data.iloc[100:120,:]
data['time'] = data.reset_index().index
data = np.array(data.Channel2)


#######################
# Persistence Diagram #
#######################
dirac = diracMaximalTimeSeries(data, k, tau)

def bettiClassic(eps):
    return persistentBettiClassic(data, k, eps, dirac, tau)

def betti(eps):
    return persistentBetti(data, k, eps, dirac, tau)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettisClassic = executor.map(bettiClassic, product(scales, reversed(scales)))

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, product(scales, reversed(scales)))


persistenceDiagram(bettisClassic, scales, figure_path='figures/diagram-eeg-classic.png', save_figure=True)

persistenceDiagram(bettis, scales, output_path='results/eeg/', figure_path='figures/diagram-eeg.png', save_data=True, save_figure=True)



