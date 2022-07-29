import numpy as np
from scipy.linalg import expm
from itertools import product
from membershipVR import *

# Create iterable with all possible k-simplices
def kcomplex(k, n):
    for simplex in product(range(2), repeat=n):
        if sum(simplex) == k:
            yield simplex

# Get coefficient of boundary matrix
def boundaryOracle(face, simplex):
    index = 0
    diff = 0
    for vf, vs in zip(face, simplex):
        if vf == 1:
            if vs == 0:
                return 0 # If face has vertex that simplex doesn't, return 0
            else:
                index += 1
        else:
            if vs == 1:
                diff += 1
                if diff > 1:
                    return 0 # If simplex has more than one extra vertex, return 0
                power = index
    return (-1)**power


# Boundary Matrix
def boundary(k, n):
    bound = []
    for face in kcomplex(k-1, n):
        row = []
        for simplex in kcomplex(k, n):
            row.append(boundaryOracle(face, simplex))
        bound.append(row)
    return np.array(bound)


# Projection Operator for Time Series
def projectionTimeSeries(data, k, n, eps, tau, d):
    proj = [membershipOracleTakens(simplex, data, eps, tau, d) for simplex in kcomplex(k, n)]
    return proj


# Projection Operator for Point Clouds
def projectionPointCloud(data, k, n, eps):
    proj = [membershipOracle(simplex, data, eps) for simplex in kcomplex(k, n)]
    return proj


# Persistent Dirac Operator for Time Series
def diracTimeSeries(data, k, eps1, eps2, tau, d, xi):
    n = data.shape
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    # Boundary Operators Restricted to Relevant Scales
    proj2 = (projectionTimeSeries(data, k, n, eps1, tau, d) > 0)
    bound1 = bound1[projectionTimeSeries(data, k-1, n, eps1, tau, d) > 0, :]
    bound1 = bound1[:, proj2]
    bound2 = bound2[proj2, :]
    proj2 = 0
    bound2 = bound2[:, projectionTimeSeries(data, k+1, n, eps2, tau, d) > 0]
    # Building Dirac Operator
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    di = np.block([
        [ (-xi) * np.eye( rows1 ) , bound1 , np.zeros( (rows1, cols2) ) ],
        [ bound1.transpose() , xi * np.eye( rows2 ) , bound2 ],
        [np.zeros( (cols2, rows1) ) , bound2.transpose() , (-xi) * np.eye( cols2 ) ],
        ])
    bound1 = 0
    bound2 = 0
    # Removing 0 rows and columns to save memory
    di = di[~np.all(di == 0, axis=1)]
    di = di[:, ~np.all(di == 0, axis=0)]
    dim, _ = di.shape
    q1 = np.ceil(np.log2(dim)) # Number of qubits for registers 1 and 2
    q1 = q1.astype(np.int64)
    if (q1 - np.log2(dim) > 0): # Filling up with 0's if necessary
        di = np.block([[di, np.zeros((dim, 2**q1 - dim))], [np.zeros((2**q1 - dim, dim)), np.zeros((2**q1 - dim, 2**q1 - dim))]])
    return (q1, di)


# Persistent Dirac Operator for Point Clouds
def diracPointCloud(data, k, eps1, eps2, xi):
    n, _ = data.shape
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    # Boundary Operators Restricted to Relevant Scales
    proj2 = (projectionPointCloud(data, k, n, eps1) > 0)
    bound1 = bound1[projectionPointCloud(data, k-1, n, eps1) > 0, :]
    bound1 = bound1[:, proj2]
    bound2 = bound2[proj2, :]
    proj2 = 0
    bound2 = bound2[:, projectionPointCloud(data, k+1, n, eps2) > 0]
    # Building Dirac Operator
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    di = np.block([
        [ (-xi) * np.eye( rows1 ) , bound1 , np.zeros( (rows1, cols2) ) ],
        [ bound1.transpose() , xi * np.eye( rows2 ) , bound2 ],
        [np.zeros( (cols2, rows1) ) , bound2.transpose() , (-xi) * np.eye( cols2 ) ],
        ])
    bound1 = 0
    bound2 = 0
    # Removing 0 rows and columns to save memory
    di = di[~np.all(di == 0, axis=1)]
    di = di[:, ~np.all(di == 0, axis=0)]
    dim, _ = di.shape
    q1 = np.ceil(np.log2(dim)) # Number of qubits for registers 1 and 2
    q1 = q1.astype(np.int64)
    if (q1 - np.log2(dim) > 0): # Filling up with 0's if necessary
        di = np.block([[di, np.zeros((dim, 2**q1 - dim))], [np.zeros((2**q1 - dim, dim)), np.zeros((2**q1 - dim, 2**q1 - dim))]])
    return (q1, di)


def UB(series, k, eps1, eps2, xi):
    dirop = dirac(series, k, eps1, eps2, xi, 1)
    m, l = parameters(data)
    M = 2**m
    ub = expm(((2*pi*l/M)*1j)*dirop)
    return ub

