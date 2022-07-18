import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import product
from classicTakens import *

##################
# Set Parameters #
##################
tau = 2 # Delay
d = 2 # Dimension of point cloud
k = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
#scales = product(range(1, 21), repeat=2)
scales = product(range(16, 19), repeat=2)


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2'])
data = data.iloc[100:150,:]
data['time'] = data.reset_index().index


#####################
# Values used often #
#####################
points = data.time.size - (tau*(d-1)) # Number of points
cloudx = data.Channel2[:points] # Point Cloud x
cloudy = data.Channel2[tau:] # Point Cloud y


############################
# Persistent Betti Numbers #
############################

def betti(eps):
    eps1, eps2 = eps
    if (eps1 > eps2):
        return 0.0
    return persistentBetti(eps1, eps2, k, points, cloudx, cloudy, xi)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, scales)

for b in bettis:
    print(b)
    
#bettis = np.asarray(bettis)


##########
# Figure #
##########
#fig, (ax1, ax2) = plt.subplots( 1, 2 )
#plt.subplots_adjust(bottom = 0.25)


###########
# Scale 1 #
###########
#ax1.bar(range(2**m10), prob1[l10 - 1, m10 - 1])
#ax1.vlines(l10*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r')
#ax1.set_title("Probability at scale 1")
#ax1.set_xlabel("p")
#ax1.set_ylabel("N x P(p)")




