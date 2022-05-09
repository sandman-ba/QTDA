#import numpy as np
#import matplotlib.pyplot as plt
#from numpy import pi
#from numpy import linalg as LA
#from math import comb as nCk
#from itertools import combinations
#from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


##############################
#        Circuit             #
##############################

qbits = QuantumRegister(8, 'q')
output = ClassicalRegister(8, 'c')
circuit = QuantumCircuit(qbits, output)

circuit.h(qubits[0])
circuit.h(qubits[1])
circuit.h(qubits[2])

circuit.h(qubits[6])
circuit.h(qubits[7])

circuit.cx(qubits[0], qubits[3])
circuit.cx(qubits[1], qubits[4])
circuit.cx(qubits[2], qubits[5])

# Exponential Dirac ##################################

circuit.QFT(qubits[6], qubits[7]) # Check notation ####################################

circuit.measure(qubits[6], output[6])
circuit.measure(qubits[7], output[7])



##############
# Parameters #
##############
#T = 9 # Number of data-times
T = 5
tau = 1 # Delay
#tau = 2
d = 2 # Dimension of point cloud
e1 = 1.0 # First scale
e2 = 1.5 # Second scale
e3 = 2.0 # Third scale
betk = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
def f(x): return np.sin((2.0*pi)*x) # Time series function


#####################
# Values used often #
#####################
points = T - (tau*(d-1)) # Number of points
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
series = f(time) # Time series
cloudx = series[:points] # Point Cloud x
cloudy = series[tau:] # Point Cloud y



