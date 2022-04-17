import numpy as np
import matplotlib as plt
import math.pi as pi

# Time series function
def f(x): return np.sin(x)

# Time series times
X = np.linespace(0.0, 2.0*pi, num=8, endpoint=False)

# Delay and dimension
tau = 1
d = 2

# Time series
Y = f(X)

# Point Cloud
V = f([X[0] , X[1]])
