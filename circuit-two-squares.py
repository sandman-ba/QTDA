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
n1 = 4
q1 = 2*n1 + m1
xi = 1.0 # Parameter for Dirac operator
s = np.sqrt(2)
xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])

##############################
#      Dirac Operators       #
##############################

dirac1 = dirac(1, 8, xcoo, ycoo, 1.1, 1.5, xi)

prob2 = 2**n1 * probp(l1, m1, dirac1)
    
##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**q1, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**q1, 2**m1))
prob1 = np.zeros((2**m1, 1))

state1 = H(q1, list( chain( range(n1), range(2*n1, 2*n1 + m1) ) ) ) @ state1
state1 = CX(q1, 0, 4) @ state1
state1 = CX(q1, 1, 5) @ state1
state1 = CX(q1, 2, 6) @ state1
state1 = CX(q1, 3, 7) @ state1
#state1 = CX(q1, 4, 11) @ state1
#state1 = CX(q1, 5, 12) @ state1
#state1 = CX(q1, 6, 13) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(2*n1, m1) @ state1

for i in range(2**m1):
    prob1[i,0] = np.vdot(state1[p1 + i, 0], state1[p1 + i, 0])

    
########################
# Plotting quantum     #
########################
fig1 = plt.figure()
plt.bar(range(2**m1), 2**n1 * prob1[:,0])
plt.title("Probability square quantum")
plt.xlabel("\(p\)")
plt.ylabel("\(\mathcal{P}(p)\)")

fig1.savefig("figures/prob-square-quantum.png")


########################
# Plotting classic     #
########################
fig2 = plt.figure()
plt.bar(range(2**m1), prob2)
plt.title("Probability square classic")
plt.xlabel("\(p\)")
plt.ylabel("\(\mathcal{P}(p)\)")

fig2.savefig("figures/prob-square-classic.png")

