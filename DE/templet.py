# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
import random


def centering(Phen):
    center = []
    for i in range(len(Phen[0])):
        gene = Phen[:, i]
        center.append(sum(gene) / len(gene))
    return center


class soea_DE_currentToBest_1_L_templet(ea.SoeaAlgorithm):
    """
soea_DE_currentToBest_1_L_templet : class - 差分进化DE/current-to-best/1/bin算法类

算法描述:
    为了实现矩阵化计算，本算法类采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
    5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。

参考文献:
    [1] Das, Swagatam & Suganthan, Ponnuthurai. (2011). Differential Evolution:
        A Survey of the State-of-the-Art.. IEEE Trans. Evolutionary Computation. 15. 4-31.

"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/current-to-best/1/L'
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        # population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            # 进行差分进化操作
            r0 = np.arange(NIND)
            r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                  [r0, None, None, r_best, r0])  # 变异
            experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果



class soea_hDE_currentToBest_1_L_templet(ea.SoeaAlgorithm):
    """
soea_DE_currentToBest_1_L_templet : class - 差分进化DE/current-to-best/1/bin算法类

算法描述:
    为了实现矩阵化计算，本算法类采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
    5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。

参考文献:
    [1] Das, Swagatam & Suganthan, Ponnuthurai. (2011). Differential Evolution:
        A Survey of the State-of-the-Art.. IEEE Trans. Evolutionary Computation. 15. 4-31.

"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/current-to-best/1/L'
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        # population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            # 进行差分进化操作
            r0 = np.arange(NIND)
            r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
                                                  [r0, None, None, r_best, r0])  # 变异
            experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群

            """Apply LS based on ES between elite population"""
            sort_index = np.argsort(-np.array(population.FitnV[:, 0]))

            elite_size = int(NIND * 0.05)
            elite_index = sort_index[0:elite_size]
            elite_Pop = population[elite_index]
            LSPop = ea.Population(population.Encoding, population.Field, 1)  # 存储LS个体
            LSPop.Chrom = np.array([centering(elite_Pop.Phen)])
            LSPop.Phen = LSPop.Chrom
            self.call_aimFunc(LSPop)

            tPop = population + LSPop
            tPop.FitnV = ea.scaling(tPop.ObjV, tPop.CV, self.problem.maxormins)  # 计算适应度
            sort_index = np.argsort(-np.array(tPop.FitnV[:, 0]))
            population = tPop[sort_index[0:NIND]]
            # population.shuffle()

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果


# def LS_generator(mean, scale, NIND):
#     lb = []
#     ub = []
#     for i in range(len(scale[0])):
#         delta = (scale[1][i] - scale[0][i]) / 2
#         lb.append(max(mean[i] - 0.1 * (delta), scale[0][i]))
#         ub.append(min(mean[i] + 0.1 * (delta), scale[1][i]))
#     samples = []
#     for i in range(NIND):
#         sample = []
#         for j in range(len(mean)):
#             sample.append((ub[j] - lb[j]) * random.random() + lb[j])
#         samples.append(sample)
#     samples.append(mean)
#     samples = np.array(samples)
#     return samples
