import numpy as np
from math import comb as nCk
from itertools import combinations, product
import membershipVR as MVR

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


# Projection
def projection(data, k, n, eps, tau=None, d=2):
    proj = [membershipOracle(simplex, data, eps) for simplex in kcomplex(k, n)]
    return proj


# Persistent Dirac Operator
def dirac(k, n, x, y, eps1, eps2, xi, mode=0):
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    proj1 = projection(k-1, n, x, y, eps1)
    proj2 = projection(k, n, x, y, eps1)
    proj3 = projection(k+1, n, x, y, eps2)
    bound1 = bound1[proj1 > 0, :]
    bound1 = bound1[:, proj2 > 0]
    bound2 = bound2[proj2 > 0, :]
    bound2 = bound2[:, proj3 > 0]
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    di = np.block([
        [ (-xi)*np.eye( rows1 ) , bound1 , np.zeros( (rows1, cols2) ) ],
        [ bound1.transpose() , (xi)*np.eye( rows2 ) , bound2 ],
        [np.zeros( (cols2, rows1) ) , bound2.transpose() , (-xi)*np.eye( cols2 ) ],
        ]) # Dirac operator
    di = di[~np.all(di == 0, axis=1)]
    di = di[:, ~np.all(di == 0, axis=0)]
    dim, _ = di.shape
    n1 = np.ceil(np.log2(dim))
    n1 = n1.astype(np.int64)
    if (n1 - np.log2(dim) > 0):
        di = np.block([[di, np.zeros((dim, 2**n1 - dim))], [np.zeros((2**n1 - dim, dim)), np.zeros((2**n1 - dim, 2**n1 - dim))]])
    if (mode > 0.5):
        return di
    return (n1, di)

def UB(series, k, eps1, eps2, xi):
    dirop = dirac(series, k, eps1, eps2, xi, 1)
    m, l = parameters(data)
    M = 2**m
    ub = expm(((2*pi*l/M)*1j)*dirop)
    return ub

