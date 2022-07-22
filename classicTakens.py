import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import linalg as LA
from math import comb as nCk
from itertools import combinations


# k-simplices
def kcomplex(k, n):
    comp = np.zeros( (nCk(n,k+1), n) )
    for row, combo in enumerate( np.array( list( combinations( range(n), k+1) ) ) ):
        comp[row][combo] = 1
    return comp


# Simplex diameter
def diameter(simplex, x, y, eps):
    vertx = x[simplex > 0.5] # x coordinate of vertices in simplex
    verty = y[simplex > 0.5] # y coordinate of vertices in simplex
    for a, b in zip(vertx,verty):
        for c, d in zip(vertx, verty):
            if LA.norm( np.array( [a - c, b - d] ), np.inf ) > eps: # If diameter is larger than eps return 0
                return 0
    return 1 # Otherwise return 1


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


# Persistent Betti Number
def persistentBetti(eps1, eps2, k, n, x, y, xi):
    eigen, _ = LA.eig(dirac(k, n, x, y, eps1, eps2, xi, 1))
    print(eigen)
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
    print(f"Betti for {eps1} to {eps2} with l={l} and M={M} is: {prob}")
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
