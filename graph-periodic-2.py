import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


##############
# Parameters #
##############
T = 12
tau = 2 # Delay
d = 2 # Dimension of point cloud
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
fig1, ax1 = plt.subplots(1, 1, figsize = (6.5, 5))
time2 = np.linspace(0.0, 2.0, 100)
ax1.plot(time2, f(time2), '-')
ax1.plot(time, series, 'o')
#ax1.set_title("Time Series")
ax1.set_xlabel(r"\(t\)")
ax1.set_ylabel(r"\(x(t)\)")


########################
# Plotting Point Cloud #
########################
fig2, ax2 = plt.subplots(1, 1, figsize = (6.5, 5))
ax2.plot(cloudx, cloudy, 'o')
#ax2.set_title("Point Cloud")
ax2.set_xlabel(r"\(x(t)\)")
ax2.set_ylabel(r"\(x(t + \tau)\)")

fig1.set_tight_layout(True)
fig2.set_tight_layout(True)

################
# Saving plots #
################
fig1.savefig("figures/time-series-two-period.png")
fig2.savefig("figures/point-cloud-two-period.png")


