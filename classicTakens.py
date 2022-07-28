import numpy as np
from numpy import pi
from numpy import linalg as LA
from persistentDirac import *

# Persistent Betti Number
def persistentBetti(eps1, eps2, k, n, x, y, xi):
    eigen, _ = LA.eig(dirac(k, n, x, y, eps1, eps2, xi, 1))
#    print(eigen)
    eigen2 = np.abs(eigen - xi)
    eigen2 = eigen2[eigen2 > 10**(-6)]
    l = np.ceil(1/np.amin(eigen2))
    eigen = l * eigen
    eigen2 = 0
    M = np.amax(np.abs(eigen))
    M = np.ceil(np.log2(M)) + 2
    M = 2**M
    p = l*xi
    prob = np.sum(((np.sin(pi*eigen) + (10**(-6)))/(np.sin(pi*(eigen - p)/M) + (10**(-6)/M)))**2)/(M**2)
#    print(f"Betti for {eps1} to {eps2} with l={l} and M={M} is: {prob}")
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
