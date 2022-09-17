import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from numpy import pi


#####################
#  Data processing  #
#####################
data = pd.read_csv('data/eeg-data.csv')
data = data.iloc[5857:,:]
data = data.drop(columns = ['IndexId', 'Ref1', 'Ref2', 'Ref3', 'TS1', 'TS2', 'Channel1'])
data = data.reset_index()
data = data.iloc[1110:1160, :]
data['time'] = data.reset_index().index


############################
# Plotting autocorrelation #
############################

fig1, ax1 = plt.subplots(1, 1, figsize = (6.5, 5))
plot_acf(data.Channel2, ax1, lags=20)

fig1.set_tight_layout(True)
fig1.savefig("figures/autocorrelation-eeg.png")

########################
# Plotting Time series #
########################
fig2, ax2 = plt.subplots(1, 1, figsize = (6.5, 5))
ax2.plot(data.time, data.Channel2, '-')
ax2.set_xlabel(r"\(t\)")
ax2.set_ylabel(r"\(x(t)\)")


fig2.set_tight_layout(True)

################
# Saving plots #
################
fig2.savefig("figures/time-series-eeg.png")
