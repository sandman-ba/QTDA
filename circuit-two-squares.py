import numpy as np
import matplotlib.pyplot as plt
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############
l1 = 2
m1 = 3
xi = 1.0 # Parameter for Dirac operator
s = np.sqrt(2)
xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])
#xcoo = np.array([0.0, 1.0, 1.0, 0.0])
#ycoo = np.array([0.0, 0.0, 1.0, 1.0])

##############################
#      Dirac Operators       #
##############################

#dirac1 = np.array(
#    [[-1.,  0.,  0.,  0., -1., -1.,  0.,  0.],
#     [ 0., -1.,  0.,  0.,  1.,  0., -1.,  0.],
#     [ 0.,  0., -1.,  0.,  0.,  0.,  1., -1.],
#     [ 0.,  0.,  0., -1.,  0.,  1.,  0.,  1.],
#     [-1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
#     [-1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
#     [ 0., -1.,  1.,  0.,  0.,  0.,  1.,  0.],
#     [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  1.]])

prob2 = probp(l1, m1, dirac1)
    
##############################
#     Phase estimation       #
##############################

state1 = np.zeros((2**9, 1), dtype = 'complex_')
state1[0,0] = 1
p1 = np.array(range(0,2**9, 2**m1))
prob1 = np.zeros((2**m1, 1))

#state1 = H(q1, list( chain( range(n1), range(2*n1, 2*n1 + m1) ) ) ) @ state1
state1 = H(9, [0, 1, 2, 6, 7, 8]) @ state1
state1 = CX(9, 0, 3) @ state1
state1 = CX(9, 1, 4) @ state1
state1 = CX(9, 2, 5) @ state1
state1 = UB(m1, l1, dirac1) @ state1
state1 = QFT(6, m1) @ state1

for i in range(2**m1):
    prob1[i,0] = np.vdot(state1[p1 + i, 0], state1[p1 + i, 0])

    
########################
# Plotting quantum     #
########################
fig1 = plt.figure()
plt.bar(range(2**m1), 8*prob1[:,0])
plt.ylim(0.0, 8.0)
plt.xlim(-0.5, 2**m1 + 1)
plt.title("Probability square quantum")
plt.xlabel("\(p\)")
plt.ylabel("\(\mathcal{P}(p)\)")

fig1.savefig("figures/prob-square-quantum.png")


########################
# Plotting classic     #
########################
fig2 = plt.figure()
plt.bar(range(2**m1), 8*prob2)
plt.ylim(0.0, 8.0)
plt.xlim(-0.5, 2**m1 + 1)
plt.title("Probability square classic")
plt.xlabel("\(p\)")
plt.ylabel("\(\mathcal{P}(p)\)")


fig2.savefig("figures/prob-square-classic.png")

