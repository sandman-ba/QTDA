import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import linalg as LA
from math import comb as nCk
from itertools import combinations


##############
# Parameters #
##############
#T = 9 # Number of data-times
#tau = 1 # Delay
#d = 2 # Dimension of point cloud
#eps1 = 0.1 # First scale
#eps2 = 0.4 # Second scale
#k = 1 # Dimension for Betti number
#xi = 1.0 # Parameter for Dirac operator
#def f(x): return np.sin((2.0*pi)*x) # Time series function


#####################
# Values used often #
#####################
#points = T - (tau*(d-1)) # Number of points
#time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
#series = f(time) # Time series
#cloudx = series[:points] # Point Cloud x
#cloudy = series[tau:] # Point Cloud y


########################
# Plotting Time series #
########################
#fig1 = plt.figure()
#plt.plot(time, series, 'o')
#plt.ylim(-1.2,1.2)
#plt.xlim(-0.2, 1.2)
#plt.title("Time Series", size = 24, weight = 'bold')
#plt.xlabel("time")
#plt.ylabel("sine")


########################
# Plotting Point Cloud #
########################
#fig2 = plt.figure()
#plt.plot(cloudx, cloudy, 'o')
#plt.ylim(-1.2, 1.2)
#plt.xlim(-1.2, 1.2)
#plt.title("Point Cloud", size = 24, weight = 'bold')
#plt.xlabel("x(t)")
#plt.ylabel("x(t + tau)")


################
# Saving plots #
################
#fig1.savefig("figures/time-series.png")
#fig2.savefig("figures/point-cloud.png")


#################
# Betti numbers #
#################
#dirop = dirac(k, n, eps1, eps2, xi) # Dirac operator
#eigen, _ = LA.eig( dirop ) # Eigenvalues and eigenvectors of dirac operator
#betti = np.sum( np.absolute(eigen - 1.0) < 0.001 ) # Multiplicity of eigenvalue 1
#print(f"The number of loops that persist from scale {eps1} to scale {eps2} is:\n {betti}")


##############################
#        Functions           #
##############################


# k-simplices
def kcomplex(k, n):
    comp = np.zeros( (nCk(n,k), n) )
    for row, combo in enumerate( np.array( list( combinations( range(n), k) ) ) ):
        comp[row][combo] = 1
    return comp


# Simplex diameter
def diameter(simplex, x, y, eps):
    vertx = x[simplex > 0.5] # x coordinate of vertices in simplex
    verty = y[simplex > 0.5] # y coordinate of vertices in simplex
    for a, b in zip(vertx,verty):
        for c, d in zip(vertx, verty):
            if LA.norm( np.array( [a - c, b - d] ) ) > eps: # If diameter is larger than eps return 0
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
    nonzero = np.zeros( (faces, simp) )
    for row, face in enumerate(comp1):
        for col, simplex in enumerate(comp2):
            nonzero[row,col] = isface(face, simplex)
            
    bound = np.block ([
        [ np.zeros( (faces, faces) ) , nonzero ],
        [ np.zeros( (simp, faces) ) , np.zeros( (simp, simp) ) ]
        ])
    return bound


# Projection
def projection(k, n, x, y, eps):
    comp = kcomplex(k, n) # k-simplices
    proj = np.zeros( (comp.shape[0], 1) )
    for i, simp in enumerate(comp):
        proj[i] = diameter(simp, x, y, eps)
    return proj


# Persistent Dirac Operator
def dirac(k, n, x, y, eps1, eps2, xi):
    bound1 = boundary(k, n) # k dimensional boundary matrix
    bound2 = boundary(k+1, n) # k-1 dimentional boundary matrix
    rows1, cols1 = bound1.shape
    rows2, cols2 = bound2.shape
    proj = np.block([
        [ np.diag(projection(k-1, n, x, y, eps1)) , np.zeros( (rows1, cols1) ) , np.zeros( (rows1, cols2) ) ],
        [ np.zeros( (cols1, rows1) ) , np.diag(projection(k, n, x, y, eps1)) , np.zeros( (rows2, cols2) ) ],
        [ np.zeros( (cols2, rows1) ) , np.zeros( (cols2, rows2) ) , np.diag(projection(k+1, n, x, y, eps2)) ]
        ]) # Projection operator
    di = proj * np.block([
        [ (-xi)*np.eye( (rows1, rows1) ) , bound1 , np.zeros( (rows1, cols2) ) ],
        [ bound1.transpose() , (xi)*np.eye( (rows2, rows2) ) , bound2 ],
        [np.zeros( (cols2, rows1) ) , bound2.transpose() , (-xi)*np.eye( (cols2, cols2) ) ],
        ]) * proj # Dirac operator
    return di


#################
#     Test      #
#################
#simp1 = np.array([0,1,1,1])
#simp2 = np.array([0,1,1,0])
#simp3 = np.array([1,1,1,1])
#simp4 = np.array([1,1,0,0])
xcoo = np.array([0,1,1,0])
ycoo = np.array([0,0,1,1])
epst = 1.0

#test1 = diameter(simp1, xcoo, ycoo, epst)
#print(f"Test 1 is {test1} and should be 0")
#test2 = diameter(simp2, xcoo, ycoo, epst)
#print(f"Test 2 is {test2} and should be 1")

#test3 = isface(simp2, simp1)
#print(f"Test 3 is {test3} and should be 1")
#test4 = isface(simp3, simp1)
#print(f"Test 4 is {test4} and should be 0")
#test5 = isface(simp4, simp1)
#print(f"Test 5 is {test5} and should be 0")

test6 = kcomplex(2, 4)
print(f"Test 6 is \n {test6}")

test7 = boundary(2, 4)
print(f"Test 7 is \n {test7}")

test8 = projection(2, 4, xcoo, ycoo, epst)
print(f"Test 8 is \n {test8}")


