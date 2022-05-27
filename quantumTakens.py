import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import e
from numpy import linalg as LA
from scipy.linalg import expm
from qiskit.circuit.library import QFT as qiskitQFT
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator as qiskitOperator

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
        ub = np.dot(ub, (ubma + ubmb))
    return np.kron(I, ub)

#def QFT(n, M):
#    m = 2**M
#    w = np.power(e, (-2*pi/m)*1j)
#    I = np.eye(2**n, dtype = 'complex_')
#    fm = (1/np.sqrt(m))*np.ones((m,m), dtype = 'complex_')
#    for i in range(1, m):
#        for s in range(i, m):
#            fm[i, s] = np.power(w, i*s)
#            fm[s, i] = fm[i, s]
#    qft = np.kron(I, fm)
#    return qft

def QFT(n, m):
    I = np.eye(2**n, dtype = 'complex_')
    circ = QuantumCircuit(m)
    circ.append(qiskitQFT(m), range(m))
    op = qiskitOperator(circ)
    fm = op.data
    qft = np.kron(I, fm)
    return qft


