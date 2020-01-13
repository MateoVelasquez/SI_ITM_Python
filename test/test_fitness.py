import os
import sys
import numpy as np
ruta = os.path.abspath(os.path.normpath('..'))
sys.path.append(ruta)
import alg_gen_sim.ftnss as alf  # noqa


def test_ftnss():
    coeff = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indep = np.array([[1], [2], [3]])
    indv = np.array([4, 5, 6])
    result = alf.fitness(indv, coeff, indep)
    res_ent = int(result[0])
    assert res_ent == 225
