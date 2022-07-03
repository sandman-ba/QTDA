import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############

T = 5
tau = 1 # Delay
d = 2 # Dimension of point cloud
e1 = 1.5 # Second scale
e2 = 2.0 # Third scale
betk = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
def f(x): return np.sin((2.0*pi)*x) # Time series function

l1 = 4 # Chosen to separate eigenvalues
m1 = 5 # Number of qubits for phase estimation (2**m > l*xi)

l2 = 4 # Chosen to separate eigenvalues
m2 = 5 # Number of qubits for phase estimation (2**m > l*xi)


#####################
# Values used often #
#####################

points = T - (tau*(d-1)) # Number of points
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
series = f(time) # Time series
cloudx = series[:points] # Point Cloud x
cloudy = series[tau:] # Point Cloud y


##############################
#      Dirac Operators       #
##############################

n1, dirac1 = dirac(betk, points, cloudx, cloudy, e1, e1, xi)
q1 = 2*n1 + m1
dim1 = 2**n1 # Dimension of Dirac operator
probc1 = probp(l1, m1, dirac1)


n2, dirac2 = dirac(betk, points, cloudx, cloudy, e1, e2, xi)
q2 = 2*n2 + m2
dim2 = 2**n2 # Dimension of Dirac operator
probc2 = probp(l2, m2, dirac2)


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

for i1 in range(2**m1):
    prob1[i1,0] = np.vdot(state1[p1 + i1, 0], state1[p1 + i1, 0])


state2 = np.zeros((2**q2, 1), dtype = 'complex_')
state2[0,0] = 1
p2 = np.array(range(0,2**q2, 2**m2))
prob2 = np.zeros((2**m2, 1))

state2 = H(q2, list( chain( range(n2), range(2*n2, 2*n2 + m2) ) ) ) @ state2
for i1, i2 in zip(range(n2), range(n2, 2*n2)):
    state2 = CX(q2, i1, i2) @ state2
state2 = UB(m2, l2, dirac2) @ state2
state2 = QFT(2*n2, m2) @ state2

for i2 in range(2**m2):
    prob2[i2,0] = np.vdot(state2[p2 + i2, 0], state2[p2 + i2, 0])

    
########################
# Plotting scale 1     #
########################
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 5))
ax1.bar(range(2**m1), dim1*prob1[:,0])
ax1.vlines(l1*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r') 
ax1.set_title("Scale \(\epsilon_1\)")
ax1.set_xlabel("\(p\)")
ax1.set_ylabel("\(N\mathcal{P}(p)\)")


#########################
# Plotting scale 1 to 2 #
#########################
ax2.bar(range(2**m2), dim2*prob2[:,0])
ax2.vlines(l2*xi, 0, 1, transform = ax2.get_xaxis_transform(), colors = 'r') 
ax2.set_title("Scale \(\epsilon_1\) to \(\epsilon_2\)")
ax2.set_xlabel("\(p\)")


############################
# Plotting scale 1 classic #
############################
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize = (11, 5))
ax3.bar(range(2**m1), dim1*probc1)
ax3.vlines(l1*xi, 0, 1, transform = ax3.get_xaxis_transform(), colors = 'r') 
ax3.set_title("Scale \(\epsilon_1\) classic")
ax3.set_xlabel("\(p\)")
ax3.set_ylabel("\(N\mathcal{P}(p)\)")


#################################
# Plotting scale 1 to 2 classic #
#################################
ax4.bar(range(2**m2), dim2*probc2)
ax4.vlines(l2*xi, 0, 1, transform = ax4.get_xaxis_transform(), colors = 'r') 
ax4.set_title("Scale \(\epsilon_1\) to \(\epsilon_2\) classic")
ax4.set_xlabel("\(p\)")


################
# Saving plots #
################
fig1.savefig("figures/prob-quantum.png")
fig2.savefig("figures/prob-classic.png")

