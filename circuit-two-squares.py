import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import e
from numpy import linalg as LA
from scipy.linalg import expm
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############
l1 = 2
m1 = 3
xi = 1.0 # Parameter for Dirac operator
#s = np.sqrt(2)
#xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
#ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])
xcoo = np.array([0.0, 1.0, 1.0, 0.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0])

##############################
#      Dirac Operators       #
##############################

#dirac1 = dirac(1, 8, xcoo, ycoo, 1.1, 1.5, xi)
#dirac1 = dirac(1, 4, xcoo, ycoo, 1.1, 1.1, xi)
#print(f"{dirac1}")
dirac1 = np.array(
    [[-1.,  0.,  0.,  0., -1., -1.,  0.,  0.],
     [ 0., -1.,  0.,  0.,  1.,  0., -1.,  0.],
     [ 0.,  0., -1.,  0.,  0.,  0.,  1., -1.],
     [ 0.,  0.,  0., -1.,  0.,  1.,  0.,  1.],
     [-1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
     [ 0., -1.,  1.,  0.,  0.,  0.,  1.,  0.],
     [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  1.]])

eigen, _ = LA.eig(dirac1)
print(f"{eigen}")
    
##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**9, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**9, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(9, [0, 1, 2, 3, 8, 9, 10]) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")
state1 = CX(9, 0, 3) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")
state1 = CX(9, 1, 4) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")
state1 = CX(9, 2, 5) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")
state1 = UB(m1, l1, dirac1) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")
state1 = QFT(6, m1) @ state1
print(f"{np.vdot(state1[:, 0], state1[:, 0])}")

for i in range(2**m1):
    prob1[i,0] = np.vdot(state1[p1 + i, 0], state1[p1 + i, 0])

print(f"{np.sum(prob1[:,0])}")
    
########################
# Plotting scale 1     #
########################
fig1 = plt.figure()
plt.bar(range(2**m1), 8*prob1[:,0])
plt.ylim(0.0, 8.0)
plt.xlim(-0.5, 2**m1 + 1)
plt.title("Probability 2 squares", size = 24, weight = 'bold')
plt.xlabel("lambda*l")
plt.ylabel("P(lambda*l)")

fig1.savefig("figures/prob-two-squares.png")

