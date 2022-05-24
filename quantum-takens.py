import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import e
from numpy import linalg as LA
from scipy.linalg import expm
#from math import comb as nCk
#from itertools import combinations
#from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
#from qiskit.circuit.library import QFT


##############################
#        Circuit             #
##############################

#q = QuantumRegister(8, 'q')
#c = ClassicalRegister(2, 'c')
#circuit = QuantumCircuit(q, c)

#circuit.h(q[0])
#circuit.h(q[1])
#circuit.h(q[2])

#circuit.h(q[6])
#circuit.h(q[7])

#circuit.cx(q[0], q[3])
#circuit.cx(q[1], q[4])
#circuit.cx(q[2], q[5])

# Exponential Dirac ##################################

#qft = QFT(2)
#circuit.append(qft, [q[6], q[7]])

#circuit.measure(q[6], c[0])
#circuit.measure(q[7], c[1])



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


##############################
#          Gates             #
##############################

def H(n, qbits):
    I = np.eye(2)
    H2 = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]])
    had = 1
#    if 0 in qbits:
#       had = 1.0*H2
#    else:
#        had = np.eye(2)
    for i in range(n):
        if i in qbits:
            had = np.kron(had, H2)
        else:
            had = np.kron(had, I)
    return had

def CX(n, control, qbit):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    q0 = np.array([[1, 0], [0, 0]])
    q1 = np.array([[0, 0], [0, 1]])
    cxa = 1
    cxb = 1
    for i in range(n):
        if i==control:
            cxa = np.kron(cxa, I)
            cxb = np.kron(cxb, X)
        else if i==qbit:
            cxa = np.kron(cxa, q0)
            cxb = np.kron(cxb, q1)
   return cxa + cxb

def UB(m, l, dirac):
    n, _ = dirac.shape
    I = np.eye(n)
    q0 = np.array([[1, 0], [0, 0]])
    q1 = np.array([[0, 0], [0, 1]])
    ud = expm(((2*pi*l/m)j)*dirac)
    ub = 1
    for i in range(m):
        ubma = 1
        ubmb = 1
        for s in range(m):
            if i == s:
                ubma = np.kron(ubma, q0)
                ubmb = np.kron(ubmb, q1)
            else:
                ubma = np.kron(ubma, q0 + q1)
                ubmb = np.kron(ubmb, q0 + q1)
        ubma = np.kron(I, ubma)
        ubmb = np.kron(LA.matrix_power(ud, 2**(m-1-i)), ubmb)
        ub = ub @ (ubma + ubmb)
    return np.kron(I, ub)

def QFT(n, m):
    w = np.power(e, (2*pi/m)j)
    I = np.eye(4*n)
    fm = (1/np.power(m, 1/2))*np.ones(m,m)
    for i in range(1, m):
        for s in range(i, m):
            fm[i, s] = np.power(w, i*s)
            fm[s, i] = fm[i, s]
    qft = np.kron(I, fm)
    return qft

##############################
#     Phase estimation       #
##############################



