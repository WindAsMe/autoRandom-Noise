import warnings
import numpy as np
from Grouping import Proposal, Comparison
from DE import DE
from cec2013lsgo.cec2013 import Benchmark
import matplotlib.pyplot as plt
from os import path


warnings.filterwarnings("ignore")


def write_obj(data, path):
    with open(path, 'a+') as f:
        f.write(str(data) + ', ')
        f.write('\n')
        f.close()


def draw_curve(x, data, title):
    plt.plot(x, data)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    Dim = 1000
    this_path = path.realpath(__file__)
    '''
    DE parameter initialization
    '''
    NIND = 50
    FEs = 3000000
    trail = 25
    '''
    Benchmark initialization
    '''

    for func_num in range(1, 16):
        bench = Benchmark()
        info = bench.get_info(func_num)
        scale_range = [info['lower'], info['upper']]
        func = bench.get_function(func_num)
        Max_iter = int(FEs / NIND / Dim)
        D_Max_iter = int((FEs - 30000) / NIND / Dim)

        D_obj_path = path.dirname(this_path) + "/Data/obj/D/f" + str(func_num)
        DECC_obj_path = path.dirname(this_path) + "/Data/obj/CC/f" + str(func_num)
        G_obj_path = path.dirname(this_path) + "/Data/obj/G/f" + str(func_num)
        MLCC_obj_path = path.dirname(this_path) + "/Data/obj/MLCC/f" + str(func_num)
        P1_obj_path = path.dirname(this_path) + "/Data/obj/P1/f" + str(func_num)
        P2_obj_path = path.dirname(this_path) + "/Data/obj/P2/f" + str(func_num)

        for i in range(trail):
            print("Func: ", func_num, " trial: ", i+1)

            """Conventional Comparison methods"""
            # D_groups = Comparison.DECC_D(Dim, func, scale_range, groups_num=10, max_number=100)
            # D_obj_trace = DE.CC(Dim, NIND, D_Max_iter, func, scale_range, D_groups)
            # write_obj(D_obj_trace, D_obj_path)
            #
            # DECC_groups = Comparison.DECC_G(Dim, 10, 100)
            # DECC_obj_trace = DE.CC(Dim, NIND, Max_iter, func, scale_range, DECC_groups)
            # write_obj(DECC_obj_trace, DECC_obj_path)
            #
            # P1_groups = Proposal.autoRandom(Dim)
            # P1_obj_trace = DE.CC(Dim, NIND, Max_iter, func, scale_range, P1_groups)
            # write_obj(P1_obj_trace, P1_obj_path)
            #
            # G_obj_trace = DE.GCC(Dim, NIND, Max_iter, func, scale_range)
            # write_obj(G_obj_trace, G_obj_path)
            #
            # MLCC_obj_trace = DE.MLCC(Dim, NIND, Max_iter, func, scale_range)
            # write_obj(MLCC_obj_trace, MLCC_obj_path)

            """Ultimate Proposal: auto random grouping + Gaussian Sampling"""
            P2_groups = Proposal.autoRandom(Dim)
            P2_obj_trace = DE.hCC(Dim, NIND, Max_iter, func, scale_range, P2_groups)
            write_obj(P2_obj_trace, P2_obj_path)



