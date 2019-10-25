import cma
import cma.purecma as purecma
from deap import benchmarks
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from scipy.optimize import minimize

from plot import *
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

def ackley(x):
    return benchmarks.ackley(x)[0]

ma_func=ackley


############## Test CMA-ES ###################

def launch_cmaes(center, sigma, nbeval=10000, display=True):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    es = cma.CMAEvolutionStrategy(centroid=[5.0]*100, sigma=5.0, lambda_=1000)

    ### A completer pour utiliser CMA-ES et tracer les individus générés à chaque étape avec plot_results###
    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # Objects that will compile the data
    sigma = numpy.ndarray((NGEN,1))
    axis_ratio = numpy.ndarray((NGEN,1))
    diagD = numpy.ndarray((NGEN,N))
    fbest = numpy.ndarray((NGEN,1))
    best = numpy.ndarray((NGEN,N))
    std = numpy.ndarray((NGEN,N))

    for gen in range(NGEN):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)

    return es.result[1]

def launch_cmaes_pure(center, sigma, nbeval=10000, display=True):
    es = purecma.CMAES(center, sigma)

    ### A completer pour utiliser CMA-ES et tracer les individus générés à chaque étape avec plot_results###

    return es.result[1]

launch_cmaes(100*[0,0],0.5)
