import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from quantumTakens import *
from classicTakens import probp

##############
# Parameters #
##############
xi = 1.0 # Parameter for Dirac operator

l1 = 3 # Chosen to separate eigenvalues
n1 = 3 # Number of qubits for the Dirac operator
m1 = 4 # Number of qubits for phase estimation (2**m > l*xi)

q1 = 2*n1 + m1 # Total number of qubits in the circuit
dim1 = 2**n1 # Dimension of Dirac operator

l2 = 3 # Chosen to separate eigenvalues
n2 = 4 # Number of qubits for the Dirac operator
m2 = 3 # Number of qubits for phase estimation (2**m > l*xi)

q2 = 2*n2 + m2 # Total number of qubits in the circuit
dim2 = 2**n2 # Dimension of Dirac operator

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

probc1 = probp(l1, m1, dirac1)

dirac2 = np.array(
    [[-1.,  0.,  0.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
     [-1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1., -1.,  0.],
     [ 0., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
     [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0., -1.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0., -1.]])

dirac2 = np.block([[ dirac2, np.zeros((14, 2))], [np.zeros((2, 14)), np.zeros((2, 2))]])

probc2 = probp(l2, m2, dirac2)

##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**q1, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**q1, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(q1, list( chain( range(n1), range(2*n1, 2*n1 + m1) ) ) ) @ state1
state1 = CX(q1, 0, 3) @ state1
state1 = CX(q1, 1, 4) @ state1
state1 = CX(q1, 2, 5) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(2*n1, m1) @ state1

for i1 in range(2**m1):
    prob1[i1,0] = np.vdot(state1[p1 + i1, 0], state1[p1 + i1, 0])


state2 = np.zeros((2**q2, 1), dtype = 'complex_')
state2[0,0] = 1
p2 = np.array(range(0,2**q2, 2**m2))
prob2 = np.zeros((2**m2, 1))

state2 = H(q2, list( chain( range(n2), range(2*n2, 2*n2 + m2) ) ) ) @ state2
state2 = CX(q2, 0, 4) @ state2
state2 = CX(q2, 1, 5) @ state2
state2 = CX(q2, 2, 6) @ state2
state2 = CX(q2, 3, 7) @ state2
state2 = UB(m2, l2, dirac2) @ state2
state2 = QFT(2*n2, m2) @ state2

for i2 in range(2**m2):
    prob2[i2,0] = np.vdot(state2[p2 + i2, 0], state2[p2 + i2, 0])

    
########################
# Plotting scale 1     #
########################
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 4))
ax1.bar(range(2**m1), dim1*prob1[:,0])
ax1.vlines(l1*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r') 
#plt.ylim(0.0, dim1)
#plt.xlim(-0.5, 2**m1 + 1)
ax1.set_title("Scale eps1")
ax1.set_xlabel("p")
ax1.set_ylabel("P(p)")


#########################
# Plotting scale 1 to 2 #
#########################
ax2.bar(range(2**m2), dim2*prob2[:,0])
ax2.vlines(l2*xi, 0, 1, transform = ax2.get_xaxis_transform(), colors = 'r') 
#plt.ylim(0.0, dim2)
#plt.xlim(-0.5, 2**m2 + 1)
ax2.set_title("Scale eps1 to eps2")
ax2.set_xlabel("p")
#ax2.ylabel("P(p)")


############################
# Plotting scale 1 classic #
############################
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize = (11, 4))
ax3.bar(range(2**m1), dim1*probc1)
ax3.vlines(l1*xi, 0, 1, transform = ax3.get_xaxis_transform(), colors = 'r') 
#plt.ylim(0.0, dim1)
#plt.xlim(-0.5, 2**m1 + 1)
ax3.set_title("Scale eps1 classic")
ax3.set_xlabel("p")
ax3.set_ylabel("P(p)")


#################################
# Plotting scale 1 to 2 classic #
#################################
ax4.bar(range(2**m2), dim2*probc2)
ax4.vlines(l2*xi, 0, 1, transform = ax4.get_xaxis_transform(), colors = 'r') 
#plt.ylim(0.0, dim2)
#plt.xlim(-0.5, 2**m2 + 1)
ax4.set_title("Scale eps1 to eps2 classic")
ax4.set_xlabel("p")
#ax4.set_ylabel("P(p)")


################
# Saving plots #
################
fig1.savefig("figures/prob-quantum.png")
fig2.savefig("figures/prob-classic.png")

