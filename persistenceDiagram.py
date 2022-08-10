import numpy as np
import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots(1, 1, figsize = (6.5, 5))

    cax = ax.matshow(bettis, cmap = 'Greys')
    cbar = fig.colorbar(cax)
    ax.set_xticks([(N//10)*x for x in range(10)])
    ax.set_yticks([(N//10)*x + 1 for x in range(10)])
    ax.set_xticklabels(map(round, scales[::N//10], repeat(1)))
    ax.set_yticklabels(map(round, scales[-2::-N//10], repeat(1)))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    cbar.set_label("Number of holes")

    fig.set_tight_layout(True)

    if save_figure:
        fig.savefig(figure_path)
    else:
        fig.show()

