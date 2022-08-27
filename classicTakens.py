import numpy as np
from numpy import pi
from numpy import linalg as LA
from persistentDirac import *


# Persistent Betti Number
def persistentBetti(data, k, eps, dirac=None, tau=None, d=2, xi=1, M_multiplier=7, l_multiplier=8):
    eps1, eps2 = eps
    if eps1 > eps2:
        return 0.0
    elif dirac is None:
        if tau is None:
            eigen, _ = LA.eig(diracPointCloud(data, k, eps1, eps2, xi)[1])
        else:
            eigen, _ = LA.eig(diracTimeSeries(data, k, eps1, eps2, tau, d, xi)[1])
    else:
        if tau is None:
            eigen, _ = LA.eig(diracPointCloud(data, k, eps1, eps2, xi, dirac)[1])
        else:
            eigen, _ = LA.eig(diracTimeSeries(data, k, eps1, eps2, tau, d, xi, dirac)[1])
    gap = np.abs(eigen - xi)
    gap = gap[gap > 10**(-9)]
    gap = np.amin(gap)
    l = np.maximum(np.ceil(1/gap), 3) * l_multiplier
    eigen = l * eigen
    M = np.amax(np.abs(eigen))
    M = np.ceil(np.log2(M)) + M_multiplier
    M = 2**M
    p = l*xi
    prob = np.sum(((np.sin(pi*eigen + (10**(-13))))/(np.sin(pi*(eigen - p)/M + (10**(-13)/M))))**2)/(M**2)
    return prob.real.astype(np.half)

# Persistent Betti Number
def persistentBettiClassic(data, k, eps, dirac=None, tau=None, d=2, xi=1):
    eps1, eps2 = eps
    if eps1 > eps2:
        return 0.0
    elif dirac is None:
        if tau is None:
            eigen, _ = LA.eig(diracPointCloud(data, k, eps1, eps2, xi)[1])
        else:
            eigen, _ = LA.eig(diracTimeSeries(data, k, eps1, eps2, tau, d, xi)[1])
    else:
        if tau is None:
            eigen, _ = LA.eig(diracPointCloud(data, k, eps1, eps2, xi, dirac)[1])
        else:
            eigen, _ = LA.eig(diracTimeSeries(data, k, eps1, eps2, tau, d, xi, dirac)[1])
    gap = np.abs(eigen - xi)
    return sum(gap < 10**(-9))

