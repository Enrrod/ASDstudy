# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from igraph import Graph


def wmatrix(file):
    A = np.genfromtxt(file, delimiter='  ')
    peso_max = np.amax(A)
    W = A / peso_max
    return W


def diffWmat(Wfin, Winit):
    Wdiff = Wfin - Winit
    return Wdiff


def tabla(array):
    cmap = plt.cm.bwr
    vmin = np.min(Wdiff)
    vmax = np.max(Wdiff)
    plt.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='None')
    plt.gca().invert_yaxis()
    plt.title('Max value: ' + "{0:.4f}".format(vmax) + ', Min value: ' + "{0:.4f}".format(vmin))
    plt.show()


if __name__=="__main__":
    file1 = '/home/enrique/Proyectos/ASDstudy/Data/TDsubjects/DTI/Wmatrices/meanValues/W_mean.csv'
    file2 = '/home/enrique/Proyectos/ASDstudy/Data/ASDsubjects/DTI/Wmatrices/meanValues/W_mean.csv'
    Winit = np.genfromtxt(file1, delimiter='  ')
    Wfin = np.genfromtxt(file2, delimiter='  ')
    Wdiff = diffWmat(Wfin, Winit)
    tabla(Wdiff)
