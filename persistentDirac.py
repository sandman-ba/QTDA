import numpy as np
from math import comb as nCk
from itertools import combinations
import membershipVR as MVR
import memebrshipTakensVR as MTVR

# k-simplices
def kcomplex(k, n):
    comp = np.zeros( (nCk(n,k+1), n) )
    for row, combo in enumerate( np.array( list( combinations( range(n), k+1) ) ) ):
        comp[row][combo] = 1
    return comp

# Get coefficient of boundary matrix
def isface(face, simplex):
    if (simplex.sum() - face.sum()) < 0.5:
        return 0 # Not a face if it's of higher dimension
    diff = face + simplex # Difference, i.e. get rid of repeated vertices
    if np.sum( np.absolute( diff - 1.0 ) < 0.5 ) > 1.0:
        return 0 # Not a face if difference is more than one vertex
    power = np.argmin( diff[ diff > 0.5 ] ) # Position of the missing vertex determines the sign
    return (-1)**power


# Boundary
def boundary(k, n):
    comp1 = kcomplex(k-1, n) # simplices dimension k-1
    comp2 = kcomplex(k, n) # simplices dimension k
    faces, _ = comp1.shape # number of simplices dimension k-1
    simp, _ = comp2.shape # number of simplices dimension k
    bound = np.zeros( (faces, simp) )
    for row, face in enumerate(comp1):
        for col, simplex in enumerate(comp2):
            bound[row,col] = isface(face, simplex)
    return bound


# Projection
def projection(k, n, x, y, eps):
    comp = kcomplex(k, n) # k-simplices
    proj = np.zeros( comp.shape[0] )
    for i, simp in enumerate(comp):
        proj[i] = diameter(simp, x, y, eps)
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

