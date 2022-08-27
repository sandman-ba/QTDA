import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from itertools import repeat

def persistenceDiagram(bettis, scales, figure_path=None):

    N = len(list(scales))
    patches = []
    colors = ['orange', 'blue']
    s0 = scales[0]
    s1 = scales[-1]
    sd = s1 / 10

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    ax.plot([s0 - sd, s1 + sd], [s0 - sd, s1 + sd], 'k')

    ax.set_xlim([s0 - sd, s1 + sd])
    ax.set_ylim([s0 - sd, s1 + sd])
    ax.set_xticks(scales[::N // 5])
    ax.set_yticks(scales[::N // 5])
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    fig.set_tight_layout(True)

    r = (s1 - s0) / 70

    for k, bettik in enumerate(bettis):

        #bettik = bettik.reshape((N,N))
        #bettik = bettik.T

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

        patches.append(PatchCollection(circles, facecolors = colors[k]))

    for patch in patches:
        ax.add_collection(patch)

    if figure_path is None:
        fig.show()
    else:
        fig.savefig(figure_path)
