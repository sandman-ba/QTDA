import numpy as np
from numpy import linalg as LA

# Simplex diameter
def diameter(simplex, x, y, eps):
    vertx = x[simplex > 0.5] # x coordinate of vertices in simplex
    verty = y[simplex > 0.5] # y coordinate of vertices in simplex
    for a, b in zip(vertx,verty):
        for c, d in zip(vertx, verty):
            if LA.norm( np.array( [a - c, b - d] ), np.inf ) > eps: # If diameter is larger than eps return 0
                return 0
    return 1 # Otherwise return 1

