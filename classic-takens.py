import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import linalg as LA
from math import comb as nCk

# Parameters
T = 9 # Number of data-times
tau = 1 # Delay
d = 2 # Dimension
eps1 = 0.1 # First scale
eps2 = 0.4 # Second scale

# Values used often
points = T - (tau*(d-1)) # Number of points
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
def f(x): return np.sin((2.0*pi)*x) # Time series function


# Time series
series = f(time)

# Point Cloud
cloud = np.array( [ [series[:points]] , [series[tau:]] ] )


# Plotting Time series
fig1 = plt.figure()
plt.plot(time, series, 'o')
plt.ylim(-1.2,1.2)
plt.xlim(-0.2, 1.2)
plt.title("Time Series", size = 24, weight = 'bold')
plt.xlabel("time")
plt.ylabel("sine")

# Plotting Point Cloud
fig2 = plt.figure()
plt.plot(cloud[0,:], cloud[1,:], 'o')
plt.ylim(-1.2,1.2)
plt.xlim(-1.2, 1.2)
plt.title("Point Cloud", size = 24, weight = 'bold')
plt.xlabel("x(t)")
plt.ylabel("x(t + tau)")

# Saving plots
fig1.savefig("figures/time-series.png")
fig2.savefig("figures/point-cloud.png")


# Betti numbers
#dirop = dirac(1, eps1, eps2) # Dirac operator
#eigen, vect = LA.eig( dirop ) # Eigenvalues and eigenvectors of dirac operator
#betti = np.sum( np.absolute(eigen - (1.0 + 0.0j)) < 0.001 ) # Multiplicity of eigenvalue 1
#print(f"The number of loops that persist from scale {eps1} to scale {eps2} is:\n {betti}")

##############################
#        Functions           #
##############################

# Boundary
def boundary(k):
    comp1 = kcomplex(k-1) # simplices dimension k-1
    comp2 = kcomplex(k) # simplices dimension k
    faces = comp1.size(0) # number of simplices dimension k-1
    simp = comp2.size(0) # number of simplices dimension k
    nonzero = np.zeros( (faces, simp) )
    for i in range(faces):
        for j in range(simp):
            nonzero[i,j] = isface(comp1[i,:], comp2[j,:])
            
    bound = np.block ([
        [ np.zeros( (faces, faces) ) , nonzero ],
        [ np.zeros( (simp, faces) ) , np.zeros( (simp, simp) ) ]
        ])
    return bound


# Projection
def projection(k, eps):
    comp = kcomplex(k) # k-simplices
    proj = np.diag( diameter(comp, eps) )
    return proj


# Persistent Dirac Operator
def dirac(k, eps1, eps2, xi):
    bound1 = boundary(k)
    bound2 = boundary(k+1)
    proj = np.block([
        [ projection(k-1, eps1) , np.zeros( () ) , np.zeros( () ) ],
        [ np.zeros( () ) , projection(k, eps1) , np.zeros( () ) ],
        [ np.zeros( () ) , np.zeros( () ) , projection(k+1, eps2) ]
        ])
    di = proj * np.block([
        [ (-xi)*np.eye( () ) , bound1 , np.zeros( () ) ],
        [ bound1 , (xi)*np.eye( () ) , bound2 ],
        [np.zeros( () ) , bound2 , (-xi)*np.eye( () ) ],
        ]) * proj
    return di


# Simplex diameter
def diameter(simplex, eps):
    diam = 0 # Need to get diameter of simplex #############################################
    return diam


# k-simplices
def kcomplex(k):
    comp = np.zeros( (nCk(points,k), points) ) # Write simplices as lists of 0s and 1s ####################
    return comp


# Get coefficient of boundary matrix
def isface(face, simplex):
    if (simplex.sum() - face.sum()) < 0.5:
        return 0
    diff = face + simplex
    if np.sum( np.absolute( diff - 1.0 ) < 0.5 ) > 1.0:
        return 0
    power = np.argmin( diff[ diff > 0.5 ] ) # Check power ########################################
    return (-1)**power
