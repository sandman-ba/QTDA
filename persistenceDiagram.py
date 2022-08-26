import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from itertools import repeat

def persistenceDiagram(bettis, scales, save_data=False, output_path='results/', save_figure=False, figure_path='figures/diagram.png'):
    N = len(list(scales))

    bettis = np.fromiter(bettis, np.double)
    bettis = bettis.reshape((N,N))
    bettis = bettis.T

    if save_data:
        np.savetxt(output_path + "scales.out", np.fromiter(scales, np.double))
        np.savetxt(output_path + "bettis.out", bettis)
        
    for j in range(N):
        for i in range(N - j - 1, 0, -1):
            bettis[i, j] = bettis[i, j] - bettis[i - 1, j]

    for i in range(N):
        for j in range(N - i - 1, 0, -1):
            bettis[i, j] = bettis[i, j] - bettis[i, j - 1]

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    circles = []
    r = (scales[-1] - scales[0]) / 70

    for x in range(N):
        for y in range(N - x):
            betti = bettis[y, x]
            if betti > 0.5:
                circle = Circle( (scales[x], scales[N - y -1]) , round(betti) * r)
                circles.append(circle)

    patch = PatchCollection(circles)

    ax.plot([scales[0] - 0.5, scales[-1]], [scales[0] - 0.5, scales[-1]], 'r')
    ax.add_collection(patch)

    ax.set_xlim([scales[0] - 0.5, scales[-1]])
    ax.set_ylim([scales[0] - 0.5, scales[-1]])
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    fig.set_tight_layout(True)

    if save_figure:
        fig.savefig(figure_path)
    else:
        fig.show()

