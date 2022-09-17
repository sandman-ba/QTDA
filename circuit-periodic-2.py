import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############

T = 12
tau = 2 # Delay
d = 2 # Dimension of point cloud
e1 = 1.6 # First scale
e2 = 1.8 # Second scale
betk = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function

l1 = 3 # Chosen to separate eigenvalues
m1 = 5 # Number of qubits for phase estimation (2**m > l*xi)

l2 = 3 # Chosen to separate eigenvalues
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
#fig1.savefig("figures/prob-double-quantum.png")
fig2.savefig("figures/prob-double-classic.png")

