import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from classicTakens import *

##################
# Set Parameters #
##################

T = 12
tau = 2 # Delay
d = 2 # Dimension of point cloud
e1 = 1.6 # First scale
e2 = 1.8 # Second scale
betk = 1 # Dimension for Betti number
xi = 1.0 # Parameter for Dirac operator
def f(x): return np.sin((2.0*pi)*x) + np.sin((4.0*pi)*x) # Time series function


#####################
# Slider Parameters #
#####################

l10 = 2 # Chosen to separate eigenvalues
l1max = 5
m10 = 3 # Number of qubits for phase estimation (2**m > l*xi)
m1max = 5

l20 = 1 # Chosen to separate eigenvalues
l2max = 5
m20 = 2 # Number of qubits for phase estimation (2**m > l*xi)
m2max = 5

#####################
# Values used often #
#####################

points = T - (tau*(d-1)) # Number of points
time = np.linspace(0.0, 1.0, num=T, endpoint=True) # Time series times
series = f(time) # Time series
cloudx = series[:points] # Point Cloud x
cloudy = series[tau:] # Point Cloud y


##############################
#      Dirac Operators       #
##############################

n1, dirac1 = dirac(betk, points, cloudx, cloudy, e1, e1, xi)
dim1 = 2**n1 # Dimension of Dirac operator


n2, dirac2 = dirac(betk, points, cloudx, cloudy, e1, e2, xi)
dim2 = 2**n2 # Dimension of Dirac operator

#################
# Probabilities #
#################

prob1 = []
for l1 in range(1, l1max + 1):
    list1 = []
    for m1 in range(1, m1max + 1):
        list1.append( dim1*probp(l1, m1, dirac1) )
    prob1.append(list1)

prob1 = np.asarray(prob1, dtype=object)

prob2 = []
for l2 in range(1, l2max + 1):
    list2 = []
    for m2 in range(1, m2max + 1):
        list2.append( dim2*probp(l2, m2, dirac2) )
    prob2.append(list2)

prob2 = np.asarray(prob2, dtype=object)


##########
# Figure #
##########

fig, (ax1, ax2) = plt.subplots( 1, 2 )
plt.subplots_adjust(bottom = 0.25)


###########
# Scale 1 #
###########
ax1.bar(range(2**m10), prob1[l10 - 1, m10 - 1])
ax1.vlines(l10*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r')
ax1.set_title("Probability at scale 1")
ax1.set_xlabel("p")
ax1.set_ylabel("N x P(p)")


######################
# Scale 1 to Scale 2 #
######################
ax2.bar(range(2**m20), prob2[l20 - 1, m20 - 1])
ax2.vlines(l20*xi, 0, 1, transform = ax2.get_xaxis_transform(), colors = 'r')
ax2.set_title("Probability from scale 1 to 2")
ax2.set_xlabel("p")


###########
# Sliders #
###########

# Positions for the sliders
ax_l1 = plt.axes([0.105, 0.15, 0.35, 0.03])
ax_m1 = plt.axes([0.105, 0.1, 0.35, 0.03])
ax_l2 = plt.axes([0.55, 0.15, 0.35, 0.03])
ax_m2 = plt.axes([0.55, 0.1, 0.35, 0.03])

slidel1 = Slider(
    ax_l1, "l", 1, l1max,
    valinit = l10, valstep = 1,
    initcolor = 'none'
)

slidem1 = Slider(
    ax_m1, "m", 1, m1max,
    valinit = m10, valstep = 1,
    color = 'green'
)

slidel2 = Slider(
    ax_l2, "l", 1, l2max,
    valinit = l20, valstep = 1,
    initcolor = 'none'
)

slidem2 = Slider(
    ax_m2, "m", 1, m2max,
    valinit = m20, valstep = 1,
    color = 'green'
)

####################
# Update Functions #
####################

# Function to update first graph
def update1(val):
    l1 = slidel1.val
    m1 = slidem1.val
    ax1.clear()
    ax1.bar(range(2**m1), prob1[l1 - 1, m1 - 1])
    ax1.vlines(l1*xi, 0, 1, transform = ax1.get_xaxis_transform(), colors = 'r')
    ax1.set_title("Probability at scale 1")
    ax1.set_xlabel("p")
    ax1.set_ylabel("N x P(p)")
    fig.canvas.draw_idle()

def update2(val):
    l2 = slidel2.val
    m2 = slidem2.val
    ax2.clear()
    ax2.bar(range(2**m2), prob2[l2 - 1, m2 - 1])
    ax2.vlines(l2*xi, 0, 1, transform = ax2.get_xaxis_transform(), colors = 'r')
    ax2.set_title("Probability from scale 1 to 2")
    ax2.set_xlabel("p")
    fig.canvas.draw_idle()


slidel1.on_changed(update1) # Update graphs when sliders change
slidem1.on_changed(update1)
slidel2.on_changed(update2)
slidem2.on_changed(update2)

ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04]) # Location of reset button
button = Button(ax_reset, 'Reset', hovercolor = '0.975') # Reset button

# Reset function activated by clicking reset button
def reset(event):
    slidel1.reset()
    slidem1.reset()
    slidel2.reset()
    slidem2.reset()

button.on_clicked(reset) # Reset to initial values when button is clicked



plt.show() # Show the graphs


