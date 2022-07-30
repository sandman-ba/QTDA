import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import product
from classicTakens import *


def persistenceDiagram(data, k, eps0, N, epsStep, tau=None, d=2, xi=1, save_data=False, output_path='results/', save_figure=False, figure_path='figures/diagram.png'):
    scales = [eps0 + (x * epsStep) for x in range(N)]

    def betti(eps):
        eps1, eps2 = eps
        if (eps1 > eps2):
            return 0.0
        return persistentBetti(data, k, eps1, eps2, tau, d, xi)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        bettis = executor.map(betti, product(scales, reversed(scales)))

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
    ax.set_xticklabels([''] + scales)
    ax.set_yticklabels([''] + scales[::-1])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    if save_figure:
        fig.savefig(figure_path)
    else:
        fig.show()

