import numpy as np


def autoRandom(Dim):
    var = np.random.permutation(list(range(0, Dim)))
    groups = [[var[0]]]
    for i in range(1, Dim):
        a = np.random.randint(0, len(groups) + 1)
        if a == len(groups):
            groups.append([var[i]])
        else:
            groups[a].append(var[i])
    return groups