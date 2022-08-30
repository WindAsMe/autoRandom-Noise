from Grouping import Proposal, Comparison
from DE import DE
from benchmark import Rastrigin, Rosenbrock, Dixon_Price, Ackley
from cec2013lsgo.cec2013 import Benchmark
import matplotlib.pyplot as plt
from os import path


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

    Dim = 500
    this_path = path.realpath(__file__)
    '''
    DE parameter initialization
    '''
    NIND = 50
    FEs = 750000
    trail = 15
    '''
    Benchmark initialization
    '''
    func_name = ["Rastrigin", "Rosenbrock", "Dixon_Price", "Ackley"]
    funcs = [Rastrigin, Rosenbrock, Dixon_Price, Ackley]
    scales = [[-5.12, 5.12], [-30, 30], [-10, 10], [-32.768, 32.768]]
    for func_num in range(1, len(func_name)):

        scale_range = scales[func_num]
        func = funcs[func_num]

        DG_groups, DG_cost = Comparison.DECC_DG(Dim, func)
        VIL_groups, VIL_cost = Comparison.CCVIL(Dim, func)

        Max_iter = int(FEs / NIND / Dim)
        DG_Max_iter = int((FEs - DG_cost) / NIND / Dim)
        VIL_Max_iter = int((FEs - VIL_cost) / NIND / Dim)
        D_Max_iter = int((FEs - 5000) / NIND / Dim)

        VIL_obj_path = path.dirname(this_path) + "/Data/obj/VIL/" + func_name[func_num]
        DG_obj_path = path.dirname(this_path) + "/Data/obj/DG/" + func_name[func_num]
        D_obj_path = path.dirname(this_path) + "/Data/obj/D/" + func_name[func_num]
        G_obj_path = path.dirname(this_path) + "/Data/obj/G/" + func_name[func_num]
        MLCC_obj_path = path.dirname(this_path) + "/Data/obj/MLCC/" + func_name[func_num]
        P1_obj_path = path.dirname(this_path) + "/Data/obj/P1/" + func_name[func_num]
        P2_obj_path = path.dirname(this_path) + "/Data/obj/P2/" + func_name[func_num]

        for i in range(trail):
            print("Func: ", func_num, " trial: ", i + 1)

            """Conventional Comparison methods"""
            D_groups = Comparison.DECC_D(Dim, func, scale_range, groups_num=10, max_number=100)
            D_obj_trace = DE.CC(Dim, NIND, D_Max_iter, func, scale_range, D_groups)
            write_obj(D_obj_trace, D_obj_path)

            G_obj_trace = DE.GCC(Dim, NIND, Max_iter, func, scale_range)
            write_obj(G_obj_trace, G_obj_path)

            MLCC_obj_trace = DE.MLCC(Dim, NIND, Max_iter, func, scale_range)
            write_obj(MLCC_obj_trace, MLCC_obj_path)

            VIL_obj_trace = DE.CC(Dim, NIND, VIL_Max_iter, func, scale_range, VIL_groups)
            write_obj(VIL_obj_trace, VIL_obj_path)

            DG_obj_trace = DE.CC(Dim, NIND, DG_Max_iter, func, scale_range, DG_groups)
            write_obj(DG_obj_trace, DG_obj_path)

            P1_obj_trace = DE.autoCC(Dim, NIND, Max_iter, func, scale_range)
            write_obj(P1_obj_trace, P1_obj_path)

            P2_groups = Proposal.autoRandom(Dim)
            P2_obj_trace = DE.MDECC(Dim, NIND, Max_iter, func, scale_range)
            write_obj(P2_obj_trace, P2_obj_path)



