import numpy as np
from numpy import pi
from numpy import linalg as LA
from persistentDirac import *

# Persistent Betti Number
def persistentBetti(data, k, eps1, eps2, tau=None, d=2, xi=1):
    if tau is None:
        eigen, _ = LA.eig(diracPointCloud(data, k, eps1, eps2, xi)[1])
    else:
        eigen, _ = LA.eig(diracTimeSeries(data, k, eps1, eps2, tau, d, xi)[1])
    gap = np.abs(eigen - xi)
    gap = gap[gap > 10**(-13)]
    gap = np.amin(gap)
    l = np.maximum(np.ceil(1/gap), 2)
    eigen = l * eigen
    M = np.amax(np.abs(eigen))
    M = np.ceil(np.log2(M)) + 1
    M = 2**M
    p = l*xi
    prob = np.sum(((np.sin(pi*eigen) + (10**(-13)))/(np.sin(pi*(eigen - p)/M) + (10**(-13)/M)))**2)/(M**2)
    return prob


# Probability Density for p
def probp(l, m, diracop):
    M = 2**m
    eigen, _ = LA.eig(diracop)
    n, _ = diracop.shape
    prob= np.array(range(M), dtype = 'float_')
    for i, p in enumerate(prob):
        prob[i] = np.sum(((np.sin(pi*l*eigen) + (10**(-6)))/(np.sin(pi*((l*eigen) - p)/M) + (10**(-6)/M)))**2)/(n*(M**2))
    return prob
