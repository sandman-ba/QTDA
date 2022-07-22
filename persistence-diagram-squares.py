import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import product
from classicTakens import *

##################
# Set Parameters #
##################
k = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
N = 8 # Number of scales
eps0 = 0.5 # Smallest scale
epsStep = 0.5 # Step between scales

#####################
# Values used often #
#####################
#s = np.sqrt(2)
scales = [eps0 + (x * epsStep) for x in range(N)]
xcoo = np.array([0.0, 1.0, 0.0, -1.0, 10.0, 10.0 + 2.0, 10.0, 10.0 - 2.0])
ycoo = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 4.0, 2.0])


############################
# Persistent Betti Numbers #
############################

def betti(eps):
    eps1, eps2 = eps
    if (eps1 > eps2):
        return 0.0
    return persistentBetti(eps1, eps2, k, 8, xcoo, ycoo, xi)

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
#fig, (ax1, ax2) = plt.subplots( 1, 2 )
#plt.subplots_adjust(bottom = 0.25)

#ax1.bar(range(2**m10), prob1[l10 - 1, m10 - 1])
#ax1.vlines(l10*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r')
#ax1.set_title("Probability at scale 1")
#ax1.set_xlabel("p")
#ax1.set_ylabel("N x P(p)")

plt.matshow(bettis)
plt.colorbar()
plt.show()


