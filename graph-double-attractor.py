import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from numpy import pi


##############
# Parameters #
##############
T = 5
tau = 1 # Delay
d = 2 # Dimension of point cloud
e1 = 1.5 # Second scale
e2 = 2.0 # Third scale
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function


#####################
# Values used often #
#####################
points = T - (tau*(d-1)) # Number of points
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
series = f(time) # Time series
cloudx = series[:points] # Point Cloud x
cloudy = series[tau:] # Point Cloud y


########################
# Plotting Time series #
########################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4.5))
time2 = np.linspace(0.0, 2.0, 100)
ax1.plot(time2, f(time2), '-')
ax1.plot(time, series, 'o')
ax1.set_title("Time Series")
ax1.set_xlabel("\(t\)")
ax1.set_ylabel("\(x(t)\)")


########################
# Plotting Point Cloud #
########################
ax2.plot(cloudx, cloudy, 'o')
ax2.set_title("Point Cloud")
ax2.set_xlabel("\(x(t)\)")
ax2.set_ylabel("\(x(t + \tau)\)")


################
# Saving plots #
################
fig.savefig("figures/double-attractor.png")


