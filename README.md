# ITM: Sistemas Inteligentes en Python.

Algunos ejemplos basicos de algoritmos inteligentes implementados en python, desarrollados para la asignatura Sistemas inteligentes del Instituto Tecnológico Metropolitano (ITM).

## Comenzando

### Instalación

Para ejecutar los codigos solo es necesario tener instalados los siguientes modulos python:
- numpy
- matplotlib

## Ejecución de pruebas

Dentro del repositorio se encuentran algunos algoritmos basicos que serán explicados a continuación:

#### Red neuronal simple
Esta [Red neuronal](https://github.com/MateoVelasquez/SI_ITM_Python/blob/master/simple_neuronal_network/red_neuronal.py) está desarrollada mediante funciones y matrices sin necesidad de un Framework adicional. Para las funciones de fordward propagation y backpropagation **NO** se incluyen regularizadores, optimizadores u otros tipos de funciones avanzadas que mejoran el desempeño de la red. 
La red se configura mediante la definicion de variables que se encuentran al inicio del código.  

- **PATH:** Corresponde a la ruta de ubicacion del dataset que será cargado.  
- **NEURONAS_CAPAS_OCULTAS:** Configuracion de las neuronas por capa oculta de la red a modo de lista, la dimension de esta corresponderá al numero de capas ocultas.  
- **EPOCAS:** Numero de iteraciones de la red.  
- **LEARNING_RATE:** Taza de aprendizaje de la red.
- **FN_ACTIVACION** Funcion de activacion de las capas. Solo puede configurarse una funcion general entre Sigmoide y ReLu que se apliaca para todas las capas.  

Se muestra un  ejemplo de configuración de la red con 3 capas ocultas (2 neuronas en la primeara capa, 4 neuronas en la segunda y 2 neuronas en la tercera capa), 10000 epocas de iteracion, taza de aprendizaje de 0.001 y función de activacion sigmoide.
```Python
PATH = 'dataset.txt'  
NEURONAS_CAPAS_OCULTAS = [2, 4, 2]
# Configuracion de la red
EPOCAS = 10000
LEARNING_RATE = 0.001
FN_ACTIVACION = "sigmoide"
```
Dentro de la carpeta de la red neuronal, se encuentran dos archivos TXT a modo de ejemplo de la estructura del dataset compactible con la red. En esta version, solo es posible el ingreso de multiples entradas y una sola salida. (La salida es la ultima columna en el TXT, las entradas son el resto de columnas)
