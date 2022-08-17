import geatpy as ea
import numpy as np
from DE import MyProblem, templet
from Grouping import Comparison


def init_Pop(Dim, scale_range, NIND):

    Field = ea.crtfld('RI', np.array([0] * Dim), np.array([[scale_range[0]] * Dim, [scale_range[1]] * Dim]),
                      np.array([[1] * Dim, [1] * Dim]))
    population = ea.Population('RI', Field, NIND)
    population.initChrom(NIND)
    return population.Chrom


def subChrom(Chrom, group):
    sub_chrom = []
    for i in range(len(Chrom)):
        sub_chrom.append([])
    for var in group:
        for i in range(len(Chrom)):
            sub_chrom[i].append(Chrom[i][var])
    return sub_chrom


def CC(Dim, NIND, MAX_iteration, func, scale_range, groups):
    context = np.zeros(Dim)
    popChrom = init_Pop(Dim, scale_range, NIND)
    real_iteration = 0
    Objs = []
    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            sub_chrom = subChrom(popChrom, groups[i])
            iteration = len(groups[i])
            solution = CC_Opt(func, scale_range, groups[i], context, sub_chrom, iteration)
            for var in groups[i]:
                popChrom[:, var] = solution['lastPop'].Chrom[:, groups[i].index(var)]
                context[var] = solution['Vars'][0][groups[i].index(var)]
        real_iteration += 1

        obj = []
        for var in popChrom:
            obj.append(func(var))
        Objs.append(min(obj))

        for i in range(1, len(Objs)):
            if Objs[i] > Objs[i-1]:
                Objs[i] = Objs[i-1]
    return Objs


def GCC(Dim, NIND, MAX_iteration, func, scale_range):
    context = np.zeros(Dim)
    popChrom = init_Pop(Dim, scale_range, NIND)
    real_iteration = 0
    Objs = []

    while real_iteration < MAX_iteration:
        groups = Comparison.DECC_G(Dim, 10, 100)
        for i in range(len(groups)):
            sub_chrom = subChrom(popChrom, groups[i])
            iteration = len(groups[i])
            solution = CC_Opt(func, scale_range, groups[i], context, sub_chrom, iteration)
            for var in groups[i]:
                popChrom[:, var] = solution['lastPop'].Chrom[:, groups[i].index(var)]
                context[var] = solution['Vars'][0][groups[i].index(var)]
        real_iteration += 1

        obj = []
        for var in popChrom:
            obj.append(func(var))
        Objs.append(min(obj))

        for i in range(1, len(Objs)):
            if Objs[i] > Objs[i-1]:
                Objs[i] = Objs[i-1]
    return Objs


def MLCC(Dim, NIND, MAX_iteration, func, scale_range):
    context = np.zeros(Dim)
    popChrom = init_Pop(Dim, scale_range, NIND)
    real_iteration = 0
    Objs = []

    while real_iteration < MAX_iteration:
        groups = Comparison.MLCC(Dim)
        for i in range(len(groups)):
            sub_chrom = subChrom(popChrom, groups[i])
            iteration = len(groups[i])
            solution = CC_Opt(func, scale_range, groups[i], context, sub_chrom, iteration)
            for var in groups[i]:
                popChrom[:, var] = solution['lastPop'].Chrom[:, groups[i].index(var)]
                context[var] = solution['Vars'][0][groups[i].index(var)]
        real_iteration += 1

        obj = []
        for var in popChrom:
            obj.append(func(var))
        Objs.append(min(obj))

        for i in range(1, len(Objs)):
            if Objs[i] > Objs[i-1]:
                Objs[i] = Objs[i-1]
    return Objs


def CC_Opt(benchmark, scale_range, group, context, Chrom, iteration):

    Field = ea.crtfld('RI', np.array([0] * len(group)),
                      np.array([[scale_range[0]] * len(group), [scale_range[1]] * len(group)]),
                      np.array([[1] * len(group), [1] * len(group)]))
    population = ea.Population('RI', Field, len(Chrom))
    population.Chrom = np.array(Chrom)
    population.Phen = np.array(Chrom)

    problem = MyProblem.CC_Problem(group, benchmark, scale_range, context)  # 实例化问题对象

    """===========================算法参数设置=========================="""
    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = iteration + 1
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution


def hCC(Dim, NIND, MAX_iteration, func, scale_range, groups):
    context = np.zeros(Dim)
    popChrom = init_Pop(Dim, scale_range, NIND)
    real_iteration = 0
    Objs = []
    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            sub_chrom = subChrom(popChrom, groups[i])
            iteration = len(groups[i])
            solution = hCC_Opt(func, scale_range, groups[i], context, sub_chrom, iteration)
            for var in groups[i]:
                popChrom[:, var] = solution['lastPop'].Chrom[:, groups[i].index(var)]
                context[var] = solution['Vars'][0][groups[i].index(var)]
        real_iteration += 1

        obj = []
        for var in popChrom:
            obj.append(func(var))
        Objs.append(min(obj))

        for i in range(1, len(Objs)):
            if Objs[i] > Objs[i-1]:
                Objs[i] = Objs[i-1]
    return Objs


def hCC_Opt(benchmark, scale_range, group, context, Chrom, iteration):

    Field = ea.crtfld('RI', np.array([0] * len(group)),
                      np.array([[scale_range[0]] * len(group), [scale_range[1]] * len(group)]),
                      np.array([[1] * len(group), [1] * len(group)]))
    population = ea.Population('RI', Field, len(Chrom))
    population.Chrom = np.array(Chrom)
    population.Phen = np.array(Chrom)

    problem = MyProblem.CC_Problem(group, benchmark, scale_range, context)  # 实例化问题对象

    """===========================算法参数设置=========================="""
    myAlgorithm = templet.soea_hDE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = iteration + 1
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution
