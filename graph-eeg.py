import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi


##############
# Parameters #
##############
tau = 2 # Delay
d = 2 # Dimension of point cloud


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2'])
data['time'] = data.reset_index().index

#####################
# Values used often #
#####################
points = data.time.size - (tau*(d-1)) # Number of points
cloudx = data.Channel2[:points] # Point Cloud x
cloudy = data.Channel2[tau:] # Point Cloud y


########################
# Plotting Time series #
########################
fig1, ax1 = plt.subplots(1, 1, figsize = (6.5, 5))
#ax1.plot(data.time[100:200], data.Channel2[100:200], '-')
ax1.plot(data.time[110:150], data.Channel2[110:150], '-')
ax1.plot(data.time[110:150], data.Channel2[110:150], 'o')
#ax1.set_title("Time Series")
ax1.set_xlabel(r"\(t\)")
ax1.set_ylabel(r"\(x(t)\)")


########################
# Plotting Point Cloud #
########################
fig2, ax2 = plt.subplots(1, 1, figsize = (6.5, 5))
ax2.plot(cloudx[110:150], cloudy[110:150], 'o')
#ax2.set_title("Point Cloud")
ax2.set_xlabel(r"\(x(t)\)")
ax2.set_ylabel(r"\(x(t + \tau)\)")

fig1.set_tight_layout(True)
fig2.set_tight_layout(True)

################
# Saving plots #
################
fig1.savefig("figures/time-series-eeg.png")
fig2.savefig("figures/point-cloud-eeg.png")


