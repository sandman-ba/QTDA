import numpy as np
import pandas as pd
import concurrent.futures
from itertools import product, repeat
from classicTakens import persistentBetti
from persistenceDiagram import *


##################
# Set Parameters #
##################
tau = 2 # Delay
k = 1 # Dimension for Betti number
N = 10 # Number of scales
eps0 = 10 # Smallest scale
epsStep = 1.0 # Step between scales
scales = [eps0 + (x * epsStep) for x in range(N)]


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2'])
data = data.iloc[100:150,:]
data['time'] = data.reset_index().index


#######################
# Persistence Diagram #
#######################
with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(persistentBetti, repeat(data.Channel2), repeat(k), product(scales, reversed(scales)), repeat(tau))


persistenceDiagram(bettis, scales, output_path='results/eeg/', figure_path='figures/diagram-eeg.png', save_data=True, save_figure=True)



