import numpy as np
from numpy import linalg as LA

# Distance Oracle Point Cloud
def distanceOracle(data, i, j, eps, mode=None):
    if mode is None:
        if (np.abs(data[i] - data[j]) > eps):
            return 0
        return 1
    else:
        if (LA.norm( data[:,i] - data[:,j]) > eps):
            return 0
    return 1

# Memebrship Oracle
def memebrshipOracle(simplex, data, eps, tau=None, d=2):
    if tau is None:
        for i, vi in enumerate(simplex):
            if (vi > 0.5):
                for j, vj in enumerate(simplex[i:]):
                    if (vj > 0.5):
                        if (distanceOracle(data, i, j, eps) < 0.5):
                            return 0
        return 1
    else:
        for i, vi in enumerate(simplex):
            if (vi > 0.5):
                for j, vj in enumerate(simplex[i:]):
                    if (vj > 0.5):
                        for t in range(d):
                            if (distanceOracle(data, i + (t * tau), j + (t * tau), eps) < 0.5):
                                return 0
        return 1

