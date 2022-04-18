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
cloud = np.array( [ series[:(points - 1)] , series[tau:] ] )


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
    faces = nCk(points, k) # Number of faces, dimension k-1
    simp = nCk(points, k+1) # Number of simplices, dimension k
    nonzero = np.zeros( (faces, simp) ) # Need to change this somehow ###################################
    bound = np.block ([
        [ np.zeros( (faces, faces) ) , nonzero ],
        [ np.zeros( (simp, faces) ) , np.zeros( (simp, simp) ) ]
        ])
    return bound


# Projection
def projection(k, eps):
    simp = nCk(points, k+1) # Number of simplices
    proj = np.zeros( (simp, simp) ) # Need to change this somehow #########################################
    return proj


# Persistent Dirac Operator
def dirac(k, eps1, eps2):
    bound1 = boundary(k)
    bound2 = boundary(k+1)
    proj1 = projection(k-1, eps1)
    proj2 = projection(k, eps1)
    proj3 = projection(k+1, eps2)
    di = 0 # Write equation for dirac operator ###########################################
    return di

