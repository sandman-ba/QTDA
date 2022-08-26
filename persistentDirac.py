import numpy as np
from numpy import pi
from scipy.sparse.linalg import expm, eigsh
from itertools import combinations
from membershipVR import *

# Create iterable with all possible k-simplices
def kcomplex(k, n):
    return list(combinations(range(n), k + 1))

# Get coefficient of boundary matrix
def boundaryOracle(face, simplex):
    index = 0
    diff = 0
    # If 'face' has extra vertex then it's not a face of simplex
    for vf in face:
        if vf not in simplex:
            return 0
    for vs in simplex:
        if vs in face:
            index += 1
        else:
            diff += 1
            if diff > 1:
                return 0 # If simplex has more than one extra vertex, return 0
            power = index
    return (-1)**power


# Boundary Matrix
def boundary(k, n):
    if k == 0:
        return np.ones((1,n), dtype=np.byte)
    bound = []
    for face in kcomplex(k-1, n):
        row = []
        for simplex in kcomplex(k, n):
            row.append(boundaryOracle(face, simplex))
        bound.append(row)
    return np.array(bound, dtype=np.byte)


# Projection Operator for Time Series
def projectionTimeSeries(data, k, n, eps1, tau, d=2, eps2=0):
    if eps2 < eps1:
        proj = [membershipOracleTakens(simplex, data, eps1, tau, d) for simplex in kcomplex(k, n)]
    else:
        proj = [membershipOracleTakens(simplex, data, eps1, tau, d) for simplex in kcomplex(k-1, n)]
        proj = proj + [membershipOracleTakens(simplex, data, eps1, tau, d) for simplex in kcomplex(k, n)]
        proj = proj + [membershipOracleTakens(simplex, data, eps2, tau, d) for simplex in kcomplex(k+1, n)]
    return np.array(proj)


# Projection Operator for Point Clouds
def projectionPointCloud(data, k, n, eps1, eps2=0):
    if eps2 < eps1:
        proj = [membershipOracle(simplex, data, eps1) for simplex in kcomplex(k, n)]
    else:
        proj = [membershipOracle(simplex, data, eps1) for simplex in kcomplex(k-1, n)]
        proj = proj + [membershipOracle(simplex, data, eps1) for simplex in kcomplex(k, n)]
        proj = proj + [membershipOracle(simplex, data, eps2) for simplex in kcomplex(k+1, n)]
    return np.array(proj)


# Maximal Dirac Operator
def diracMaximalTimeSeries(data, k, tau, d=2, xi=1):
    n = data.shape[0] - (tau * (d - 1))
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    # Building Dirac Operator
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    di = np.block([
        [ (-xi) * np.eye( rows1 , dtype=np.byte) , bound1 , np.zeros( (rows1, cols2) , dtype=np.byte) ],
        [ bound1.transpose() , xi * np.eye( rows2 , dtype=np.byte) , bound2 ],
        [np.zeros( (cols2, rows1) , dtype=np.byte) , bound2.transpose() , (-xi) * np.eye( cols2 , dtype=np.byte) ],
        ])
    return di.astype(np.byte)

def diracMaximalPointCloud(data, k, xi=1):
    n = data.shape[0]
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    # Building Dirac Operator
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    di = np.block([
        [ (-xi) * np.eye( rows1 ) , bound1 , np.zeros( (rows1, cols2) ) ],
        [ bound1.transpose() , xi * np.eye( rows2 ) , bound2 ],
        [np.zeros( (cols2, rows1) ) , bound2.transpose() , (-xi) * np.eye( cols2 ) ],
        ])
    return di.astype(np.byte)


# Persistent Dirac Operator for Time Series
def diracTimeSeries(data, k, eps1, eps2, tau, d=2, xi=1, dirac=None):
    n = data.shape[0] - (tau * (d - 1))
    if dirac is None:
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
            [ (-xi) * np.eye( rows1 , dtype=np.byte) , bound1 , np.zeros( (rows1, cols2) , dtype=np.byte) ],
            [ bound1.transpose() , xi * np.eye( rows2 , dtype=np.byte) , bound2 ],
            [np.zeros( (cols2, rows1) , dtype=np.byte) , bound2.transpose() , (-xi) * np.eye( cols2 , dtype=np.byte) ],
            ])
        bound1 = 0
        bound2 = 0
    else:
        proj = (projectionTimeSeries(data, k, n, eps1, tau, d, eps2) > 0)
        di = dirac[proj, :]
        di = di[:, proj]
    # Removing 0 rows and columns to save memory
    di = di[~np.all(di == 0, axis=1)]
    di = di[:, ~np.all(di == 0, axis=0)]
    dim, _ = di.shape
    q1 = np.ceil(np.log2(dim)) # Number of qubits for registers 1 and 2
    q1 = q1.astype(int)
    if (q1 - np.log2(dim) > 0): # Filling up with 0's if necessary
        di = np.block([[di, np.zeros((dim, 2**q1 - dim), dtype=np.byte)], [np.zeros((2**q1 - dim, dim), dtype=np.byte), np.zeros((2**q1 - dim, 2**q1 - dim), dtype=np.byte)]])
    return (q1, di.astype(np.byte))


# Persistent Dirac Operator for Point Clouds
def diracPointCloud(data, k, eps1, eps2, xi=1, dirac=None):
    n, _ = data.shape
    if dirac is None:
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
            [ (-xi) * np.eye( rows1 , dtype=np.byte) , bound1 , np.zeros( (rows1, cols2) , dtype=np.byte) ],
            [ bound1.transpose() , xi * np.eye( rows2 , dtype=np.byte) , bound2 ],
            [np.zeros( (cols2, rows1) , dtype=np.byte) , bound2.transpose() , (-xi) * np.eye( cols2 , dtype=np.byte) ],
            ])
        bound1 = 0
        bound2 = 0
    else:
        proj = (projectionPointCloud(data, k, n, eps1, eps2) > 0)
        di = dirac[proj, :]
        di = di[:, proj]
     # Removing 0 rows and columns to save memory
    di = di[~np.all(di == 0, axis=1)]
    di = di[:, ~np.all(di == 0, axis=0)]
    dim, _ = di.shape
    q1 = np.ceil(np.log2(dim)) # Number of qubits for registers 1 and 2
    q1 = q1.astype(int)
    if (q1 - np.log2(dim) > 0): # Filling up with 0's if necessary
        di = np.block([[di, np.zeros((dim, 2**q1 - dim), dtype=np.byte)], [np.zeros((2**q1 - dim, dim), dtype=np.byte), np.zeros((2**q1 - dim, 2**q1 - dim), dtype=np.byte)]])
    return (q1, di.astype(np.byte))


def UB(data, k, eps1, eps2, tau=None, d=2, xi=1, M_multiplier=2):
    if tau is None:
        q1, ub = diracPointCloud(data, k, eps1, eps2, xi)
    else:
        q1, ub = diracTimeSeries(data, k, eps1, eps2, tau, d, xi)
    gap = eigsh(ub - (xi * np.eye(2**q1)), 2**(q1-1), sigma=0.000001, which='LM', return_eigenvectors=False)
    gap = gap[gap > 0]
    gap = np.amin(gap)
    l = np.maximum(np.ceil(1/gap), 2)
    m = eigsh(ub, 1, which='LM', return_eigenvectors=False)[0] * l
    m = np.ceil(np.log2(m)) + M_multiplier
    M = 2**m
    ub = expm(((2*pi*l/M)*1j)*ub)
    return (l, m, q1, ub)

