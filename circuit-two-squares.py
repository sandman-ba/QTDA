import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############
l1 = 3
m1 = 5
xi = 1.0 # Parameter for Dirac operator
s = np.sqrt(2)
xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])

##############################
#      Dirac Operators       #
##############################

n1, dirac1 = dirac(1, 8, xcoo, ycoo, 1.1, 1.5, xi)
q1 = 2*n1 + m1

prob2 = 2**n1 * probp(l1, m1, dirac1)
    
##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**q1, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**q1, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(q1, list( chain( range(n1), range(2*n1, 2*n1 + m1) ) ) ) @ state1
for i1, i2 in zip(range(n1), range(n1, 2*n1)):
    state1 = CX(q1, i1, i2) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(2*n1, m1) @ state1

for i in range(2**m1):
    prob1[i,0] = np.vdot(state1[p1 + i, 0], state1[p1 + i, 0])

    
########################
# Plotting quantum     #
########################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 5))
ax1.bar(range(2**m1), 2**n1 * prob1[:,0])
ax1.vlines(l1*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r') 
ax1.set_title("Probability squares quantum")
ax1.set_xlabel("\(p\)")
ax1.set_ylabel("\(N\mathcal{P}(p)\)")

########################
# Plotting classic     #
########################
ax2.bar(range(2**m1), prob2)
ax2.vlines(l1*xi, 0, 1, transform = ax2.get_xaxis_transform(), colors = 'r') 
ax2.set_title("Probability squares classic")
ax2.set_xlabel("\(p\)")

fig.savefig("figures/prob-squares.png")

