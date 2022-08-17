import random
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def base_fitness(Population, func, intercept):
    fitness = []
    for indi in Population:
        fitness.append(func(indi) - intercept)
    return fitness


def group_individual(group, individual):
    part_individual = np.zeros(len(individual))
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(groups, Population, func, cost, intercept):
    fitness = []
    for indi in Population:
        indi_fitness = 0
        for group in groups:
            indi_fitness += (func(group_individual(group, indi)) - intercept)
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, opt_fitness):
    error = 0
    for i in range(len(base_fitness)):
        error += ((base_fitness[i] - opt_fitness[i])) ** 2
    return error


benchmark = Benchmark()


class Function:
    def __init__(self, func_num):
        self.func_num = func_num

    def get_func(self):
        if self.func_num == 1:
            return f1
        elif self.func_num == 2:
            return f2
        elif self.func_num == 3:
            return f3
        elif self.func_num == 4:
            return f4
        elif self.func_num == 5:
            return f5
        elif self.func_num == 6:
            return f6
        elif self.func_num == 7:
            return f7
        elif self.func_num == 8:
            return f8
        elif self.func_num == 9:
            return f9
        elif self.func_num == 10:
            return f10
        elif self.func_num == 11:
            return f11
        else:
            return None

    def get_info(self):
        func_info = benchmark.get_info(self.func_num)
        return [func_info['lower'], func_info['upper']]


"""
Part 1: Fully separable functions
"""


# f1 + f1
def f1(solution):
    func = benchmark.get_function(1)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f2 + f2
def f2(solution):
    func = benchmark.get_function(2)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f3 + f3
def f3(solution):
    func = benchmark.get_function(3)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


"""
Part 2: Partially separable functions I
"""


# f4 + f4
def f4(solution):
    func = benchmark.get_function(4)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f5 + f5
def f5(solution):
    func = benchmark.get_function(5)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


"""
Part 3: Partially separable functions II
"""


# f8 + f8
def f6(solution):
    func = benchmark.get_function(8)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f9 + f9
def f7(solution):
    func = benchmark.get_function(9)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


"""
Part 4: overlapping functions
"""


# f12 + f12
def f8(solution):
    func = benchmark.get_function(12)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f13 + f13
def f9(solution):
    func = benchmark.get_function(13)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f14 + f14
def f10(solution):
    func = benchmark.get_function(14)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))


# f15 + f15
def f11(solution):
    func = benchmark.get_function(15)
    return func(np.array(solution[0:1000], dtype='double')) + func(np.array(solution[1000:2000], dtype='double'))
