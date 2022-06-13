import numpy as np
import matplotlib.pyplot as plt
from quantumTakens import *
from classicTakens import *

##############
# Parameters #
##############
xi = 1.0 # Parameter for Dirac operator
s = np.sqrt(2)
xcoo = np.array([0.0, 1.0, 1.0, 0.0, 10.0, 10.0 + s, 10.0 + s, 10.0])
ycoo = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, s, s])
#xcoo = np.array([0.0, 1.0, 1.0, 0.0])
#ycoo = np.array([0.0, 0.0, 1.0, 1.0])

##############################
#      Dirac Operators       #
##############################

dirac1 = dirac(1, 8, xcoo, ycoo, 1.1, 1.5, xi)
#dirac1 = dirac(1, 4, xcoo, ycoo, 1.1, 1.1, xi)
print(f"{dirac1}")
print(f"{dirac1.shape}")
