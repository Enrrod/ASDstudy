# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from deap import creator, base, tools, algorithms
from sklearn.metrics import mean_squared_error as mse
from scoop import futures
from time import time, sleep


def wmatrix(file):
    A = np.genfromtxt(file, delimiter='  ')
    peso_max = np.amax(A)
    W = A / peso_max
    return W

def degmatrix(W):
    tam = W.shape[0]
    d = np.zeros((tam, 1))
    I = np.identity(tam)
    for i in range(tam):
        for j in range(tam):
            d[i] = d[i] + W[i, j]
    D = I * d
    return D


def Lmatrix(W, D):
    Lnt = D - W
    Lt = Lnt.transpose()
    L = np.dot(Lt, Lnt)
    L = L / 2
    return L


def eigen(L):
    (LANDA, PHY) = np.linalg.eig(L)  # calculamos los autovalores
    tam = PHY.shape[1]
    ind = np.argsort(LANDA)  # de menor a mayor
    #ind = ind[::-1]         # de mayor a menor
    LANDA = LANDA[ind]
    for i in range(tam):
        PHY[i] = PHY[i][ind]
    return PHY, LANDA


def eigen_reduce(PHY, LANDA):
    del_ind = np.where(LANDA <= 0)[0]
    LANDA_ses = np.delete(LANDA, del_ind)
    PHY_ses = np.delete(PHY, del_ind, 1)
    return PHY_ses, LANDA_ses


def eigen_aisle(PHY_ses):
    # Aislamos el 9º autovector
    phy9 = np.empty(0)
    for i in range(PHY_ses.shape[0]):
        phy9 = np.hstack((phy9, PHY_ses[i, 7].real))
    return phy9


def fit_function(individual, phy_mean):
    degmat = degmatrix(individual)
    lmat = Lmatrix(individual, degmat)
    (phys, landas) = eigen(lmat)
    phys_ses, landa_ses = eigen_reduce(phys, landas)
    phy9 = eigen_aisle(phys_ses)
    # Aplicamos el MSE (Error Cuadrático Medio) como función a minimizar
    error_phy = mse(phy_mean, phy9)
    return error_phy,


def matMutFloat(individual, rowindpb, elemindpb, mask):
    size = len(individual)
    for i in range(size):
        rowindMut = random.random()
        if rowindMut < rowindpb:
            for j in range(size):
                elemindMut = random.random()
                if elemindMut < elemindpb:
                    attrMut = random.random()
                    individual[i][j], individual[j][i] = attrMut, attrMut
    np.place(individual, mask, 0)
    return individual,


def patchCx(ind1, ind2):
    n = len(ind1)
    tam = np.random.randint(1, (n / 2) + 1)
    patch1 = ind1[0:tam,(n-tam):n].copy()
    patch2 = ind2[0:tam, (n - tam):n].copy()
    ind1[0:tam, (n - tam):n], ind1[(n - tam):n, 0:tam] = patch2, patch2.T
    ind2[0:tam, (n - tam):n], ind2[(n - tam):n, 0:tam] = patch1, patch1.T
    del patch1
    del patch2
    return ind1, ind2,


def obtainMask(file):
    mask = wmatrix(file)
    mask = np.where(mask > 0, 0, 1)
    return mask


def graphInd(icls, dim, mask):
    indGenerator = np.random.rand(dim, dim)
    graphInd = (indGenerator + indGenerator.T) / 2
    np.place(graphInd, mask, 0)
    return icls(graphInd)


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('individual', graphInd, creator.Individual, dim=264,
                 mask=obtainMask('/home/enrique/Proyectos/ASDstudy/Data/ASDsubjects/DTI/Wmatrices/ASD38D_DTI_connectivity_matrix_file.txt'))
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', fit_function,
                 phy_mean=np.genfromtxt('/home/enrique/Proyectos/ASDstudy/Data/TDsubjects/DTI/Wmatrices/meanValues/phy9_mean.csv', delimiter=','))
toolbox.register('mate', patchCx)
toolbox.register('mutate', matMutFloat, rowindpb=0.1, elemindpb=0.1,
                 mask=obtainMask(
                     '/home/enrique/Proyectos/ASDstudy/Data/ASDsubjects/DTI/Wmatrices/ASD38D_DTI_connectivity_matrix_file.txt'))
toolbox.register('select', tools.selTournament, tournsize=3)
#toolbox.register("map", futures.map)


def main():
    mutpb = 0.1
    cxpb = 0.6
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('avg', np.mean, axis=0)

    logbook = tools.Logbook()
    population = toolbox.population(100)
    NGEN = 10
    print("Starting optimization with " + str(NGEN) + " generations")
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        top = tools.selBest(population, k=1)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print("Generation (" + str(gen + 1) + "/" + str(NGEN) + ") completed")

    return logbook, top


if __name__ == "__main__":
    init_time = time()
    file = '/home/enrique/Proyectos/ASDstudy/Data/ASDsubjects/DTI/Wmatrices/ASD38D_DTI_connectivity_matrix_file.txt'
    phy_mean = np.genfromtxt('/home/enrique/Proyectos/ASDstudy/Data/TDsubjects/DTI/Wmatrices/meanValues/phy9_mean.csv',
                             delimiter=',')
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #toolbox.register("map", pool.map)
    logbook, top = main()
    #pool.close()
    generation = logbook.select('gen')
    fitness_min = logbook.select('min')
    fitness_avg = logbook.select('avg')
    top_ind = top[0]

    time = time() - init_time
    print("Execution time: " + str(time))

    w_pat = wmatrix(file)
    init_error = fit_function(w_pat, phy_mean)
    init_error = init_error[0]
    fitness_rel = (fitness_min / init_error) * 100
    fitness_avg_rel = (fitness_avg / init_error) * 100

    plt.figure()
    line1 = plt.plot(generation, fitness_rel, "b-", label="Relative fitness")
    line2 = plt.plot(generation, fitness_avg_rel, "r-", label="Relative average Fitness", alpha=0.5)
    plt.axis([0, len(generation), -10, 120])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    lines1 = [line1[0], line2[0]]
    labs1 = [line1[0].get_label(), line2[0].get_label()]
    plt.legend(lines1, labs1, loc="upper right")
    plt.title("cxpb= 0.6 mutpb= 0.1")
    plt.show()

