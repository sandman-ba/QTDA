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
e2 = 1.5 # Second scale
e3 = 2.0 # Third scale
betk = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator


##############################
#      Dirac Operators       #
##############################

#dirac1 =
#dirac2 = 

##############################
#          Gates             #
##############################

def H(n, qbits):
    I = np.eye(2)
    H2 = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]])
    had = 1
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
            cxa = np.kron(cxa, q0)
            cxb = np.kron(cxb, q1)
        elif i==qbit:
            cxa = np.kron(cxa, I)
            cxb = np.kron(cxb, X)
        else:
            cxa = np.kron(cxa, I)
            cxb = np.kron(cxb, I)
    return cxa + cxb

def UB(m, l, dirac):
    n, _ = dirac.shape
    M = 2**m
    I = np.eye(n)
    q0 = np.array([[1, 0], [0, 0]])
    q1 = np.array([[0, 0], [0, 1]])
    ud = expm(((2*pi*l/M)*1j)*dirac)
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

def QFT(n, M):
    m = 2**M
    w = np.power(e, (2*pi/m)*1j)
    I = np.eye(2**n, dtype = 'complex_')
    fm = (1/np.sqrt(m))*np.ones((m,m), dtype = 'complex_')
    for i in range(1, m):
        for s in range(i, m):
            fm[i, s] = np.power(w, i*s)
            fm[s, i] = fm[i, s]
    qft = np.kron(I, fm)
    return qft

##############################
#     Phase estimation       #
##############################



##############################
#           Tests            #
##############################

#test1 = H(2, [1])
#test2 = H(2, [0])
#test3 = H(3, [0, 2])

#print(f"{test1}")
#print(f"{test2}")
#print(f"{test3}")


#test4 = CX(2, 1, 0)
#test5 = CX(2, 0, 1)
#test6 = CX(3, 0, 1)
#test7 = CX(3, 0, 2)

#print(f"{test4}")
#print(f"{test5}")
#print(f"{test6}")
#print(f"{test7}")

#test8 = QFT(0,1)
#test9 = QFT(1,1)
#test10 = QFT(0,2)
#test11 = QFT(1,2)

#print(f"{test8}")
#print(f"{test9}")
#print(f"{test10}")
#print(f"{test11}")
