import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import product
from classicTakens import *


##################
# Set Parameters #
##################
k = 1 # Dimension for Betti number
N = 20 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.1 # Step between scales


#####################
# Values used often #
#####################
s = np.sqrt(2)
scales = [eps0 + (x * epsStep) for x in range(N)]
#data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [10.0, 0.0], [10.0 + s, 0.0], [10.0 + s, s], [10.0, s]])


############################
# Persistent Betti Numbers #
############################
def betti(eps):
    eps1, eps2 = eps
    if (eps1 > eps2):
        return 0.0
    return persistentBetti(data, k, eps1, eps2)

with concurrent.futures.ProcessPoolExecutor() as executor:
    bettis = executor.map(betti, product(scales, reversed(scales)))

bettis = np.fromiter(bettis, np.double)
bettis = bettis.reshape((N,N))
bettis = bettis.T

np.savetxt("results/squares/scales.out", np.fromiter(scales, np.double))
np.savetxt("results/squares/bettis.out", bettis)


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
ax.set_title("Persistence Diagram Squares")
ax.set_xlabel("Birth")
ax.set_ylabel("Death")

fig.savefig("figures/diagram-squares.png")

