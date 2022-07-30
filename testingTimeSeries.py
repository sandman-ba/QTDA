import numpy as np
from numpy import pi
from numpy import linalg as LA
from classicTakens import *


T = 5 # Number of points
k = 1 # Dimension for Betti number
tau = 1 # Delay
e1 = 1.5 # Second scale
e2 = 2.0 # Third scale
def f(x): return np.sin((2.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series


#print(data)

#print(distanceOracle(data, 2, 3, eps1))
#print(membershipOracle((0, 0, 1, 1), data, eps1))

#for simplex in kcomplex(k, n):
#    print(simplex)
#    print(membershipOracle(simplex, data, eps1))
#    for face in kcomplex(k-1,n):
#        print(face)
#        print(boundaryOracle(face, simplex))

#print(boundary(k, n))
#print(boundary(k+1, n))


#print(projectionPointCloud(data, k-1, n, eps1))
#print(projectionPointCloud(data, k, n, eps1))
#print(projectionPointCloud(data, k+1, n, eps2))

#bound1 = boundary(k, n)
#bound2 = boundary(k+1, n)
#proj1 = projectionPointCloud(data, k-1, n, eps1)
#proj2 = projectionPointCloud(data, k, n, eps1)
#proj3 = projectionPointCloud(data, k+1, n, eps2)
#bound1 = bound1[proj1 > 0, :]
#bound1 = bound1[:, proj2 > 0]
#bound2 = bound2[proj2 > 0, :]
#bound2 = bound2[:, proj3 > 0]

#print(bound1)
#print(bound2)


#print(diracPointCloud(data, k, eps1, eps2, 1)[1])

#l, m, q1, ub = UB(data, k, eps1, eps2)
#print(l)
#print(m)
#print(q1)
#print(ub)

#print(persistentBetti(data, k, eps1, eps1))
#print(persistentBetti(data, k, eps1, eps2))



