import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from itertools import repeat

def persistenceDiagram(bettis, scales, figure_path=None):

    N = len(list(scales))
    patches = []

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    ax.plot([scales[0] - 0.5, scales[-1]], [scales[0] - 0.5, scales[-1]], 'r')

    ax.set_xlim([scales[0] - 0.5, scales[-1]])
    ax.set_ylim([scales[0] - 0.5, scales[-1]])
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    fig.set_tight_layout(True)

    r = (scales[-1] - scales[0]) / 70

    for bettik in bettis:

        bettik = bettik.reshape((N,N))
        bettik = bettik.T

        for j in range(N):
            for i in range(N - j - 1, 0, -1):
                bettik[i, j] = bettik[i, j] - bettik[i - 1, j]

        for i in range(N):
            for j in range(N - i - 1, 0, -1):
                bettik[i, j] = bettik[i, j] - bettik[i, j - 1]

        circles = []

        for x in range(N):
            for y in range(N - x):
                betti = bettik[y, x]
                if betti > 0.5:
                    circle = Circle( (scales[x], scales[N - y -1]) , round(betti) * r)
                    circles.append(circle)

        patches.append(PatchCollection(circles))

    for patch in patches:
        ax.add_collection(patch)

    if figure_path is None:
        fig.show()
    else:
        fig.savefig(figure_path)
