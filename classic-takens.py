import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

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
V = f([X[:-(tau*d)] , X[tau:]])

# Plotting Time series
fig1 = plt.figure()
plt.plot(X, Y, 'o')
plt.ylim(-1.2,1.2)
plt.xlim(-0.2, 6.5)
plt.title("Time Series", size = 24, weight = 'bold')
plt.xlabel("time")
plt.ylabel("sine")


# Plotting Point Cloud
fig2 = plt.figure()
plt.plot(V[0,:], V[1,:], 'o')
plt.ylim(-1.2,1.2)
plt.xlim(-1.2, 1.2)
plt.title("Point Cloud", size = 24, weight = 'bold')
plt.xlabel("x(t)")
plt.ylabel("x(t + tau)")

# Saving plots
fig1.savefig("figures/time-series.png")
fig2.savefig("figures/point-cloud.png")


