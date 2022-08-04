import numpy as np
import matplotlib.pyplot as plt


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

    fig, ax = plt.subplots(1, 1)

    cax = ax.matshow(bettis)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + scales[::1])
    ax.set_yticklabels([''] + scales[::-1])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    if save_figure:
        fig.savefig(figure_path)
    else:
        fig.show()

