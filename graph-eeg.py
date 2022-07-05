import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from numpy import pi


##############
# Parameters #
##############
tau = 2 # Delay
d = 2 # Dimension of point cloud


#####################
# Values used often #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2'])
data['time'] = data.reset_index().index

#####################
# Values used often #
#####################
T = data.time.size()
points = T - (tau*(d-1)) # Number of points
cloudx = data.Chanel1[:points] # Point Cloud x
cloudy = data.Chanel1[tau:] # Point Cloud y


########################
# Plotting Time series #
########################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4.5))
ax1.plot(data.time, data.Chanel1, '-')
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
fig.savefig("figures/eeg.png")


