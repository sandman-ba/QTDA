import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import product
from classicTakens import *

##################
# Set Parameters #
##################
T = 5 # Number of points
k = 1 # Dimension for Betti number
tau = 1 # Delay
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales

#####################
# Values used often #
#####################
scales = [eps0 + (x * epsStep) for x in range(N)]
def f(x): return np.sin((2.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series


############################
# Persistent Betti Numbers #
############################

def betti(eps):
    eps1, eps2 = eps
    if (eps1 > eps2):
        return 0.0
    return persistentBetti(data, k, eps1, eps2, tau)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, product(scales, reversed(scales)))

bettis = np.fromiter(bettis, np.double)
bettis = bettis.reshape((N,N))
bettis = bettis.T

np.savetxt("results/one-period/scales.out", np.fromiter(scales, np.double))
np.savetxt("results/one-period/bettis.out", bettis)


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
ax.set_xticklabels([''] + scales[::4])
ax.set_yticklabels([''] + scales[::-4])
ax.xaxis.set_ticks_position('bottom')
ax.set_title("Persistence Diagram One Period")
ax.set_xlabel("Birth")
ax.set_ylabel("Death")

#plt.show()
fig.savefig("figures/diagram-one-period.png")



