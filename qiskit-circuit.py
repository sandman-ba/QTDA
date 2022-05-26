#import numpy as np
#import matplotlib.pyplot as plt
#from numpy import pi
#from numpy import e
#from numpy import linalg as LA
#from scipy.linalg import expm

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QFT


##############################
#        Circuit             #
##############################

q = QuantumRegister(8, 'q')
c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(q, c)

circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])

circuit.h(q[6])
circuit.h(q[7])

circuit.cx(q[0], q[3])
circuit.cx(q[1], q[4])
circuit.cx(q[2], q[5])

# Exponential Dirac ##################################
# UnitaryGate function?

qft = QFT(2)
circuit.append(qft, [q[6], q[7]])

circuit.measure(q[6], c[0])
circuit.measure(q[7], c[1])

print(circuit)

