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
l1 = 3
m1 = 2
xi = 1.0 # Parameter for Dirac operator
s = np.sqrt(2)
xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])

##############################
#      Dirac Operators       #
##############################

dirac1 = dirac(1, 8, xcoo, ycoo, 0.8, 1.5, xi)

##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**10, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**10, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(10, [0, 1, 2, 3, 8, 9]) @ state1
state1 = CX(10, 0, 4) @ state1
state1 = CX(10, 1, 5) @ state1
state1 = CX(10, 2, 6) @ state1
state1 = CX(10, 3, 7) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(8, m1) @ state1

for i in range(2**m1):
    prob1[i,0] = np.vdot(state1[p1 + i, 0], state1[p1 + i, 0])

    
########################
# Plotting scale 1     #
########################
fig1 = plt.figure()
plt.bar(range(2**m1), 12*prob1[:,0])
plt.ylim(0.0, 12.0)
plt.xlim(-0.5, 2**m1 + 1)
plt.title("Probability 2 squares", size = 24, weight = 'bold')
plt.xlabel("lambda*l")
plt.ylabel("P(lambda*l)")

fig1.savefig("figures/prob-two-squares.png")

