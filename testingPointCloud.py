import numpy as np
from persistentDirac import *
from membershipVR import *
from classicTakens import *
from itertools import combinations

k = 1 # Dimension for Betti number
n = 4
eps1 = 1.1
eps2 = 2.0
s = np.sqrt(2)
#data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [10.0, 0.0], [10.0 + s, 0.0], [10.0 + s, s], [10.0, s]])
data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
#data = np.array([[0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0],[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s]])
#data = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

print(data)

#print(distanceOracle(data, 2, 3, eps1))
#print(membershipOracle((0, 0, 1, 1), data, eps1))

for simplex in kcomplex(k, n):
    print(simplex)
    print(membershipOracle(simplex, data, eps1))
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

#print(persistentBetti(data, k, (eps1, eps1)))
#print(persistentBetti(data, k, (eps1, eps2)))
