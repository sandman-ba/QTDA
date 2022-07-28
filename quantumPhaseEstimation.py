from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QFT


##############################
#        Circuit             #
##############################

q1 = QuantumRegister(n, 'q1')
q2 = QuantumRegister(n, 'q2')
R = QuantumRegister(m, 'R')
c = ClassicalRegister(m, 'c')
circuit = QuantumCircuit(q1, q2, R, c)

circuit.h(q1)
circuit.h(q2)

for i in range(n):
    circuit.cx(q1[i], q2[i])

# Exponential Dirac ##################################
# UnitaryGate function?

circuit.append(QFT(m, inverse=True), R)

circuit.measure(R, c)

print(circuit)

