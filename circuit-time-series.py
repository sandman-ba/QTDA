import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import e
from numpy import linalg as LA
from scipy.linalg import expm
from quantumTakens import *

##############
# Parameters #
##############
l1 = 2
l2 = 1
m1 = 3
m2 = 2
xi = 1.0 # Parameter for Dirac operator


##############################
#      Dirac Operators       #
##############################

dirac1 = np.array(
   [[-1.,  0.,  0.,  0., -1., -1.,  0.,  0.],
    [ 0., -1.,  0.,  0.,  1.,  0., -1.,  0.],
    [ 0.,  0., -1.,  0.,  0.,  0.,  1., -1.],
    [ 0.,  0.,  0., -1.,  0.,  1.,  0.,  1.],
    [-1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
    [-1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
    [ 0., -1.,  1.,  0.,  0.,  0.,  1.,  0.],
    [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  1.]])

dirac2 = np.array(
    [[-1.,  0.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0., -1.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0., -1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
     [-1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
     [-1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0., -1., -1.,  0.],
     [ 0., -1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.],
     [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
     [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0., -1.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0., -1.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0., -1.]])

dirac2 = np.block([[ dirac2, np.zeros((12, 4))], [np.zeros((4, 12)), np.zeros((4, 4))]])


##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**9, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**9, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(9, [0, 1, 2, 6, 7, 8]) @ state1
state1 = CX(9, 0, 3) @ state1
state1 = CX(9, 1, 4) @ state1
state1 = CX(9, 2, 5) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(6, m1) @ state1

for i1 in range(2**m1):
    prob1[i1,0] = np.vdot(state1[p1 + i1, 0], state1[p1 + i1, 0])


state2 = np.zeros((2**10, 1), dtype = 'complex_')
state2[0,0] = 1
p2 = np.array(range(0,2**10, 2**m2))
prob2 = np.zeros((2**m2, 1))

state2 = H(10, [0, 1, 2, 3, 8, 9]) @ state2
state2 = CX(10, 0, 4) @ state2
state2 = CX(10, 1, 5) @ state2
state2 = CX(10, 2, 6) @ state2
state2 = CX(10, 3, 7) @ state2
state2 = UB(m2, l2, dirac2) @ state2
state2 = QFT(8, m2) @ state2

for i2 in range(2**m2):
    prob2[i2,0] = np.vdot(state2[p2 + i2, 0], state2[p2 + i2, 0])

    
########################
# Plotting scale 1     #
########################
fig1 = plt.figure()
plt.bar(range(2**m1), 8*prob1[:,0])
plt.ylim(0.0, 8.0)
plt.xlim(-0.5, 2**m1 + 1)
plt.title("Probability at e1", size = 24, weight = 'bold')
plt.xlabel("lambda*l")
plt.ylabel("P(lambda*l)")


#########################
# Plotting scale 1 to 2 #
#########################
fig2 = plt.figure()
plt.bar(range(2**m2), 12*prob2[:,0])
plt.ylim(0.0, 12.0)
plt.xlim(-0.5, 2**m2 + 1)
plt.title("Probability from e1 to e2", size = 24, weight = 'bold')
plt.xlabel("lambda*l")
plt.ylabel("P(lambda*l)")


################
# Saving plots #
################
fig1.savefig("figures/prob-e1.png")
fig2.savefig("figures/prob-e1-e2.png")

