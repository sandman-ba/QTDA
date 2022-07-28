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
N = 10 # Number of scales
eps0 = 10 # Smallest scale
epsStep = 1.0 # Step between scales


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
scales = [eps0 + (x * epsStep) for x in range(N)]
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
    bettis = executor.map(betti, product(scales, reversed(scales)))

bettis = np.fromiter(bettis, np.double)
bettis = bettis.reshape((N,N))
bettis = bettis.T

np.savetxt("results/eeg/scales.out", np.fromiter(scales, np.double))
np.savetxt("results/eeg/bettis.out", bettis)


######################
# Persistent Diagram #
######################

for j in range(N):
    for i in range(N - j - 1, 0, -1):
        bettis[i, j] = bettis[i, j] - bettis[i - 1, j]

for i in range(N):
    for j in range(N - i - 1, 0, -1):
        bettis[i, j] = bettis[i, j] - bettis[i, j - 1]


##########
# Figure #
##########
fig, ax = plt.subplots(1, 1)

cax = ax.matshow(bettis)
fig.colorbar(cax)
ax.set_xticklabels([''] + scales[::2])
ax.set_yticklabels([''] + scales[::-2])
ax.xaxis.set_ticks_position('bottom')
ax.set_title("Persistence Diagram EEG")
ax.set_xlabel("Birth")
ax.set_ylabel("Death")

#plt.show()
fig.savefig("figures/diagram-eeg.png")


