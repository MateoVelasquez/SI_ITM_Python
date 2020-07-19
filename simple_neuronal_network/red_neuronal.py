""" Código de red neuronal.

Ejemplo de programacion de una red neuronal por medio de matrices.
"""
import numpy as np
import sys
from matplotlib import pyplot as plt


# ---------- CONFIGURACION DE ESTRUCTURA Y ENTRENAMIENTO-------------

# # Datos de entrada: características y etiquetas
# Ruta del Dataset. Esta version del codigo solo soporta una salida
# y multiples entradas
PATH = 'datasetf.txt'

# # Definición de la red (neuronas por capa oculta)
# Se definen las neuronas por capa OCULTA (numeros enteros),
# la dimension de la lista corresponderá al numero de capas ocultas.
NEURONAS_CAPAS_OCULTAS = [2, 4, 2]


# Configuracion de la red
EPOCAS = 10000
LEARNING_RATE = 0.001
FN_ACTIVACION = "sigmoide"

# Las funciones de activacion disponibles son "sigmoide" o "relu"

# ------------------------- INICIO DEL CODIGO------------------------


def sigmoid(z, derivate=False):
    """Definicion de la funcion sigmoide.

    Creacion de la funcion de sigmoide 1/(1+(e^-z))

    Parameters
    ----------
    z: tuple
        Dato a evaluar con la funcion sigmoide.
    derivate: Bool
        False: La funcion a evaluar es sigmoide(z).
        True: La funcion a evaluar es la derivada de sigmoide(z)

    Returns
    -------
    resultado: Float
        Resultado de evaluar la funcion sigmoide.
    """
    def sig_fn(z):
        return 1 / (1 + np.exp(-z))

    if derivate is False:
        resultado = sig_fn(z)
    else:
        resultado = sig_fn(z)*(1.0-sigmoid(z))
    return resultado


def relu(z, derivate=False):
    """Definicion de la funcion relu.

    Creacion de la funcion relu f(x) = max(0,x)

    Parameters
    ----------
    z: tuple
        Dato a evaluar con la funcion ReLu.
    derivate: Bool
        False: La funcion a evaluar es ReLu(z).
        True: La funcion a evaluar es la derivada de ReLu(z)

    Returns
    -------
    relu_fn:
        Resultado de evaluar la funcion ReLu.
    """
    if derivate is False:
        relu_fn = np.maximum(0, z)
    else:
        x = z.copy()
        x[x <= 0] = 0
        x[x > 0] = 1
        relu_fn = x
    return relu_fn


def initizalizeParameters(n_in, n_hl, n_out):
    """Función para incializar parametros.

    Recibe la definicion de la red, las entradas y las salidas,
    devuelve un diccionario de parametros que contiene la estructura
    inicializada de la red.

    Parameters
    ----------
    nx: ndarray
        Arreglo que corresponde a la entrada de la red.
    n_hl: List
        Lista que contiene el numero de neuronas por capa oculta.
    n_out: ndarray
        Arreglo que corresponde a la salida de la red (clases).

    Returns
    -------
    Parametros: Dict
        Diccionario que contiene los parametros de pesos y biases de la red.
    """
    parametros = {}
    n_hl.append(int(n_out))
    for idx, val in enumerate(n_hl):
        if idx == 0:
            w_i = np.random.rand(val, n_in)
        else:
            w_i = np.random.rand(val, n_hl[idx-1])
        b_i = np.zeros((val, 1))
        parametros[f'w{idx}'] = w_i
        parametros[f'b{idx}'] = b_i
    return parametros


def forward_propagation(x, y, parametros, fn_act):
    """ Propagacion de los datos de entrada.

    Toma los datos de entrada y los pasa a través de la red.

    Parameters
    ----------
    x: ndarray
        Datos de entrada para la red.
    y: ndarray
        Datos que deben salir de la red según las entradas.
    parametros: Dict
        Estructura inicializada de la red.
    fn_act: String
        Funcion de activacion para la capa.

    Returns
    -------
    cost: Float
        Resultado de la funcion de error de la red.
    cache: Dict
        Lista de valores resultantes entre capas de la red.
    a_i: ndarray
        Arreglo de salida de la red en su ultima capa.
    """
    cache = {}
    out_activated = [x]
    # Pasa las entradas a travez de toda la red.
    # El siguiente for itera entre las capas.
    for i in range(0, len(parametros)//2):
        # Recuperacion de parametros
        w_i = parametros[f'w{i}']
        b_i = parametros[f'b{i}']
        # Producto punto de pesos entre capas y funcion de activacion
        z_i = np.dot(w_i, out_activated[i]) + b_i
        if fn_act == 'sigmoide':
            a_i = sigmoid(z_i)
        elif fn_act == 'relu':
            a_i = relu(z_i)
        out_activated.append(a_i)
        # Almacenamiento de resultados
        cache[f'z_{i}'] = z_i
        cache[f'a_{i}'] = a_i
        cache[f'w_{i}'] = w_i
        cache[f'b_{i}'] = b_i
    # Cálculo de entropia cruzada
    logprobs = np.multiply(y, np.log(a_i)) + np.multiply(np.log(1-a_i), (1-y))
    cost = - np.sum(logprobs)/x.shape[1]
    # https://stats.stackexchange.com/questions/167787/cross-entropy-cost-function-in-neural-network
    return cost, cache, a_i


def backward_propagation(x, y, cache, fn_act='sigmoide'):
    """Propagación hacia atrás.

    Propaga el error hacia atrás para el cálculo del gradiente.
    La funcion backpropagacion fue cambiada desde la original, a su forma
    matricial, siguiendo la explicacion de la siguiente fuente.:
    https://sudeepraja.github.io/Neural/

    Parameters
    ----------
    x: ndarray
        Datos de entrada.
    y: ndarray
        Datos de salida.
    cache: Dict
        Diccionario que contiene los datos de la propagacion hacia adelante.
    fn_act: String
        Funcion de activacion para las capas. Deafult:'sigmoide'

    Returns
    -------
    grad: Dict
        Diccionario que contiene los gradientes de los pesos para las capas.
    """
    cache['a_-1'] = x
    errxlayer = {}
    grad = {}
    # recuperacion de valores desde la caché:
    layers = len(cache.keys())//4 - 1
    for lyr in range(layers, -1, -1):
        if fn_act == 'sigmoide':
            term2 = sigmoid(np.dot(cache[f'w_{lyr}'], cache[f'a_{lyr-1}']))
        elif fn_act == 'relu':
            term2 = relu(np.dot(cache[f'w_{lyr}'], cache[f'a_{lyr-1}']))
        if lyr == layers:
            term1 = (cache[f'a_{lyr}'] - y)
        else:
            term1 = np.dot(cache[f'w_{lyr+1}'].T, errxlayer[f'E_{lyr+1}'])
        errxlayer[f'E_{lyr}'] = np.multiply(term1, term2)
        grad[f'dE_dw{lyr}'] = np.dot(errxlayer[f'E_{lyr}'],
                                     cache[f'a_{lyr-1}'].T)
        grad[f'dB{lyr}'] = np.sum(errxlayer[f'E_{lyr}'], axis=0, keepdims=True)
    return grad


def update_parameters(parameters, grads, learning_rate):
    """Actualizar parametros

    Actualiza los parametros segun la funcion gradiente y el indice
    de aprendizaje.

    Parameters
    ----------
    parameters: Dict
        Diccionario que contiene los valores de pesos y los biases para
        cada capa.
    grads: Dict
        Diccionario que contiene la informacion de los gradientes.
    learning_rate: Float
        Taza de aprendizaje de la red.

    Returns
    -------
    new_parameters: Dict
        Diccionario con los nuevos parametros para la Red.
    """
    layers = len(parameters.keys())//2
    new_parameters = {}
    for lyr in range(0, layers):
        new_parameters[f'w{lyr}'] = (parameters[f'w{lyr}'] -
                                     learning_rate * grads[f'dE_dw{lyr}'])
        new_parameters[f'b{lyr}'] = (parameters[f'b{lyr}'] -
                                     learning_rate * grads[f'dB{lyr}'])
    return new_parameters


def carga_dataset(path):
    """Carga de dataset

    Funcion para cargar el dataset desde un archivo TXT.
    El dataset debe estar separado por columnas (separadas por espacios)
    y por filas, separadas con cambio de linea.
    La salida de la red será la ultima fila (esta funcion de carga
    solo soporta una salida)
    La entrada de la red serán el resto de columnas.

    Parameters
    ----------
    path: String
        Direccion de donde se encuentra el archivo TXT

    Returns
    -------
    insarray, outsarray: ndarray, ndarray
        Entradas y salidas en formato numpy array para que la red pueda leerla
    """
    dset = np.loadtxt(path, comments="#", delimiter=" ", unpack=False)
    nrow, ncols = dset.shape
    outs = []
    ins = []
    for col in range(ncols):
        vec_col = []
        for row in range(nrow):
            if col == ncols-1:
                outs.append(dset[row][-1])
            else:
                vec_col.append(dset[row][col])
        ins.append(vec_col)
    ins = ins[0:-1]
    outs = [outs]
    insarray = np.array(ins)
    outsarray = np.array(outs)
    print(f'Numero de entradas: {insarray.shape[0]}')
    print(f'Numero de salidas: {outsarray.shape[0]}')
    print(f'Numero de parametros: {insarray.shape[1]}')
    return insarray, outsarray


if __name__ == "__main__":

    # cargando dataset
    try:
        DATOS_IN, DATOS_OUT = carga_dataset(PATH)
    except Exception as e:
        print(e)
        print('\n')
        sys.exit('No se pudo cargar el Dataset o está dañado.'
                 'Por favor revise las entradas.')

    # Inicializando red
    neuronas_in = DATOS_IN.shape[0]
    neuronas_out = DATOS_OUT.shape[0]
    parameters = initizalizeParameters(neuronas_in,
                                       NEURONAS_CAPAS_OCULTAS,
                                       neuronas_out)
    losses = np.ones((EPOCAS, 1))
    acc = (np.zeros((EPOCAS, 1), dtype=float))
    grap_loses = [1]
    grap_acc = [0]

    # Entrenar
    for epoca in range(EPOCAS):
        losses[epoca, 0], cache, red_out = forward_propagation(DATOS_IN,
                                                               DATOS_OUT,
                                                               parameters,
                                                               FN_ACTIVACION)
        gradientes = backward_propagation(DATOS_IN, DATOS_OUT,
                                          cache, FN_ACTIVACION)
        parameters = update_parameters(parameters, gradientes, LEARNING_RATE)
        acc[epoca, 0] = round(np.mean(1-(DATOS_OUT - red_out)**2), 3)
        grap_acc.append(acc[epoca, 0])
        grap_loses.append(losses[epoca, 0])
        imp = (f'Epoca: {epoca} | Perdida: {round(losses[epoca, 0], 3)}.'
               f' Precision: {acc[epoca, 0]} |  \n--------\r')
        sys.stdout.write(imp)
        sys.stdout.flush()
        if losses[epoca, 0] <= 0.05:
            losess = losses[0:epoca]
            break

    # Predecir
    costo, _, salida = forward_propagation(DATOS_IN, DATOS_OUT,
                                           parameters, FN_ACTIVACION)
    # Graficar
    plt.figure()
    plt.plot(grap_loses[1::], label='Perdida')
    plt.plot(grap_acc[1::], label='Precision')
    plt.legend()
    plt.title('Desempeño de la Red')
    plt.xlabel('Epocas')
    plt.show()
