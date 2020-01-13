"""Módulo para el calculo del ftnss

"""


def fitness(indv, mtx_coeff, mtx_indp):
    trans = indv.reshape(-1, 1)
    aux1 = mtx_coeff@trans
    aux2 = abs(mtx_indp-aux1)
    result = sum(aux2)
    return result


def fitness_full():
    pass


def fitness_genes():
    pass
