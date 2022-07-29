import numpy as np
from numpy import linalg as LA

# Distance Oracle Point Cloud
def distanceOracle(data, i, j, eps):
    if (LA.norm( data[:,i] - data[:,j]) > eps):
        return 0
    return 1

# Distance Oracle Time Series
def distanceOracleTakens(data, t, s, eps):
    if (np.abs(data[t] - data[s]) > eps):
        return 0
    return 1


# Memebrship Oracle for Point Clouds
def memebrshipOracle(simplex, data, eps):
    for i, vi in enumerate(simplex):
        if vi == 1:
            for j, vj in enumerate(simplex[i:]):
                if vj == 1:
                    if distanceOracle(data, i, j, eps) == 0:
                        return 0
    return 1

# Memebrship Oracle for Time Series
def memebrshipOracleTakens(simplex, data, eps, tau, d):
    for i, vi in enumerate(simplex):
        if vi == 1:
            for j, vj in enumerate(simplex[i:]):
                if vj == 1:
                    for t in range(d):
                        if distanceOracle(data, i + (t * tau), j + (t * tau), eps) == 0:
                            return 0
    return 1

