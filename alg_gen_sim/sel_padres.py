"""Módulo para seleccioar padres.

Este modulo devuelve una matriz con los individuos padres
aleatorios a partir de una probabilidad deterministica y una poblacion

"""
import random


def extpadres(prob, pobla):
    """Extrae 2 padres de la muestra.

    Se selecciona aleatoriamente un numero de
    individuos de una población y se extraen los dos
    mejores para obtener los padres.

    Parameters
    ----------
    prob: int
        Numero de individuos a seleccionar de la poblacion.
    pobla: List
        Poblacion de individuos.

    Returns
    -------
    mtx: List
        Matriz de individuos padre.

    """
    ran_elem = random.sample(range(0, len(pobla)), prob)
    mtx_padres = []
    for elem in ran_elem:
        mtx_padres.append(pobla[elem])
    return(mtx_padres)
