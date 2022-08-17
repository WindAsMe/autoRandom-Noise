import numpy as np
import random
import geatpy as ea
from DE import templet


def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups


def DECC_DG(N, f):
    cost = 2
    groups = CCDE(N)
    intercept = f(np.zeros(N)) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

    for i in range(len(groups)-1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros(N)
            index1[groups[i][0]] = 1
            delta1 = f(index1) - intercept

            for j in range(i+1, len(groups)):
                cost += 2
                if i < len(groups)-1 and j < len(groups) and not DG_Differential(N, groups[i][0], groups[j][0], delta1, f, intercept):
                    groups[i].extend(groups.pop(j))
                    j -= 1
    return groups, cost


def DG_Differential(Dim, e1, e2, a, func, intercept):
    index1 = np.zeros(Dim)
    index2 = np.zeros(Dim)
    index1[e2] = 1
    index2[e1] = 1
    index2[e2] = 1

    b = func(index1) * (1 + np.random.normal(loc=0, scale=0.01, size=None)) - intercept
    c = func(index2) * (1 + np.random.normal(loc=0, scale=0.01, size=None)) - intercept

    return np.abs(c - (a + b)) < 0.001


def DECC_G(Dim, groups_num=10, max_number=100):
    return k_s(Dim, groups_num, max_number)



def DECC_D(Dim, func, scale_range, groups_num=20, max_number=50):

    NIND = Dim * 10
    delta = OptTool(Dim, NIND, func, scale_range)
    groups_index = list(np.argsort(delta))
    groups = []
    for i in range(groups_num):
        group = groups_index[i * max_number: (i + 1) * max_number]
        groups.append(group)
    return groups


class MyProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.benchmark = benchmark
        maxormins = [-1]
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)


def OptTool(Dim, NIND, f, scale_range):
    problem = MyProblem(Dim, f, scale_range)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    population.Phen = population.Chrom
    problem.aimFunc(population)
    """===========================算法参数设置=========================="""
    Initial_Chrom = population.Chrom
    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 2
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    Optimized_Chrom = solution["lastPop"].Chrom
    delta = []
    for i in range(Dim):
        delta.append(abs(sum(Optimized_Chrom[:, i]) - sum(Initial_Chrom[:, i])))
    return delta


def MLCC(Dim):
    sizes = [5, 10, 25, 50]
    size = sizes[np.random.randint(0, len(sizes)-1)]
    return DECC_G(Dim, groups_num=int(size), max_number=int(Dim / size))


def k_s(Dim, groups_num=20, max_number=50):
    groups = []
    groups_index = list(range(Dim))
    random.shuffle(groups_index)
    for i in range(groups_num):
        group = groups_index[i * max_number: (i+1) * max_number]
        groups.append(group)
    return groups



def CCVIL(Dim, func):
    cost = 2
    groups = CCDE(Dim)
    f0 = func(np.zeros(Dim)) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

    for i in range(len(groups) - 1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros(Dim)
            index1[groups[i][0]] = 1
            fi = func(index1) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

            for j in range(i + 1, len(groups)):
                cost += 2
                if i < len(groups) - 1 and j < len(groups) and not Monotonicity_check(Dim, groups[i][0], groups[j][0], fi,
                                                                                      func, f0):
                    groups[i].extend(groups.pop(j))
                    j -= 1
    return groups, cost


def Monotonicity_check(Dim, e1, e2, fi, func, f0):
    index1 = np.zeros(Dim)
    index2 = np.zeros(Dim)
    index1[e2] = 1
    index2[e1] = 1
    index2[e2] = 1

    fj = func(index1) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
    fij = func(index2) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

    return (fij > fj > f0 and fij > fi > f0) or (fij < fj < f0 and fij < fi < f0)
