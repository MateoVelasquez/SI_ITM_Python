import os
import sys
ruta = os.path.abspath(os.path.normpath('..'))
sys.path.append(ruta)
import alg_gen_sim.sel_padres as ags


def test_selectpadres_3gen():
    prob_deter = 3
    test_pob = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15]]
    elem = ags.extpadres(prob_deter, test_pob)
    print(elem)


def test_selectpadres_5gen():
    prob_deter = 8
    test_pob = [[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]]
    print(prob_deter)
    print(test_pob)


test_selectpadres_3gen()
