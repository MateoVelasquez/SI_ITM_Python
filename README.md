# ITM: Sistemas Inteligentes en Python.

Algunos ejemplos basicos de algoritmos inteligentes implementados en python, desarrollados para la asignatura Sistemas inteligentes del Instituto Tecnológico Metropolitano (ITM).

## Comenzando

### Instalación

Para ejecutar los codigos solo es necesario tener instalados los siguientes modulos python:
- numpy
- matplotlib

## Ejecución de pruebas

Dentro del repositorio se encuentran algunos algoritmos basicos que serán explicados a continuación:

### Red neuronal simple
Esta [Red neuronal](https://github.com/MateoVelasquez/SI_ITM_Python/blob/master/simple_neuronal_network/red_neuronal.py) está desarrollada mediante funciones y matrices sin necesidad de un Framework adicional.
La red se configura mediante la definicion de variables que se encuentran al inicio del código:
**PATH:** Corresponde a la ruta de ubicacion del dataset que será cargado.
```Python
PATH = 'datasetf.txt'  
NEURONAS_CAPAS_OCULTAS = [2, 4, 2]


# Configuracion de la red
EPOCAS = 10000
LEARNING_RATE = 0.001
FN_ACTIVACION = "sigmoide"

```
Dentro de la carpeta, se encuentran dos archivos TXT a modo de ejemplo de la estructura del dataset compactible con la red. En esta version, solo
