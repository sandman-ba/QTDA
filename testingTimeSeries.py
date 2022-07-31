import numpy as np
from numpy import pi
from classicTakens import *
from membershipVR import *
from persistentDirac import *

T = 5 # Number of points
n = 4
k = 1 # Dimension for Betti number
tau = 1 # Delay
d = 2
eps1 = 1.5 # Second scale
eps2 = 2.5 # Third scale
def f(x): return np.sin((2.0*pi)*x) # Time series function
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
data = f(time) # Time series


print(data)

#print(distanceOracleTakens(data, 2, 3, eps1))
#print(membershipOracleTakens((0, 0, 1, 1), data, eps1, tau, d))

#for simplex in kcomplex(k, n):
#    print(simplex)
#    print(membershipOracleTakens(simplex, data, eps1, tau, d))
#    for face in kcomplex(k-1,n):
#        print(face)
#        print(boundaryOracle(face, simplex))

#print(boundary(k, n))
#print(boundary(k+1, n))


#print(projectionTimeSeries(data, k-1, n, eps1, tau, d))
#print(projectionTimeSeries(data, k, n, eps1, tau, d))
#print(projectionTimeSeries(data, k+1, n, eps2, tau, d))

#bound1 = boundary(k, n)
#bound2 = boundary(k+1, n)
#proj1 = projectionTimeSeries(data, k-1, n, eps1, tau, d)
#proj2 = projectionTimeSeries(data, k, n, eps1, tau, d)
#proj3 = projectionTimeSeries(data, k+1, n, eps2, tau, d)
#bound1 = bound1[proj1 > 0, :]
#bound1 = bound1[:, proj2 > 0]
#bound2 = bound2[proj2 > 0, :]
#bound2 = bound2[:, proj3 > 0]

#print(bound1)
#print(bound2)


#print(diracTimeSeries(data, k, eps1, eps2, tau, d, 1)[1])

#l, m, q1, ub = UB(data, k, eps1, eps2, tau)
#print(l)
#print(m)
#print(q1)
#print(ub)

print(persistentBetti(data, k, (eps1, eps1), tau))
print(persistentBetti(data, k, (eps1, eps2), tau))



