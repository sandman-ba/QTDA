import numpy as np
from numpy import linalg as LA

# Distance Oracle Point Cloud
def distanceOracle(data, i, j, eps):
    if (LA.norm( data[i, :] - data[j, :]) > eps):
        return 0
    return 1

# Distance Oracle Time Series
def distanceOracleTakens(data, t, s, eps):
    if (np.abs(data[t] - data[s]) > eps):
        return 0
    return 1


# Memebrship Oracle for Point Clouds
def membershipOracle(simplex, data, eps):
    for i, vi in enumerate(simplex):
        for vj in enumerate(simplex[i:]):
            if distanceOracle(data, vi, vj, eps) == 0:
                return 0
    return 1

# Memebrship Oracle for Time Series
def membershipOracleTakens(simplex, data, eps, tau, d):
    for i, vi in enumerate(simplex):
        for vj in enumerate(simplex[i:]):
            for t in range(d):
                if distanceOracleTakens(data, vi + (t * tau), vj + (t * tau), eps) == 0:
                    return 0
    return 1

