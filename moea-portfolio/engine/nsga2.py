import numpy as np
import random
from deap import base, creator, tools, algorithms
from engine.objectives import evaluate
from engine.constraints import repair_weights

def setup_nsga2(mean_returns, cov_matrix, returns_df, n_stocks,
                pop_size=100, n_gen=200):
    # Safe creator registration
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=n_stocks)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(ind):
        repair_weights(ind)
        return evaluate(ind, mean_returns, cov_matrix, returns_df)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate",   tools.cxSimulatedBinaryBounded,
                     low=0, up=1, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=0, up=1, eta=20.0, indpb=1.0 / n_stocks)
    toolbox.register("select", tools.selNSGA2)
    return toolbox, pop_size, n_gen

def run_nsga2(toolbox, pop_size, n_gen,
              cx_prob=0.9, mut_prob=0.1, progress_callback=None):
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    # Evaluate initial population
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))
    hof.update(pop)

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values
        for ind in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(ind)
                del ind.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)
        if progress_callback:
            progress_callback(gen, n_gen)

    return pop, hof
