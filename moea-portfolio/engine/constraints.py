import numpy as np

def repair_weights(individual):
    w = np.clip(np.array(individual), 0, 1)
    s = w.sum()
    w = np.ones(len(w)) / len(w) if s == 0 else w / s
    for i in range(len(individual)):
        individual[i] = w[i]
    return individual
