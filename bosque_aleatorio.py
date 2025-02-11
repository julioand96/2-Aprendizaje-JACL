import random
from arboles_numericos import entrena_arbol, NodoN, predice_arbol

def crea_subconjuntos(datos, M):
    """
    Crea M subconjuntos de datos con selección aleatoria con repetición.
    
    Parámetros:
    -----------
    datos: list(dict)
        Lista de diccionarios donde cada diccionario representa una instancia.
    M: int
       Número de subconjuntos a crear.
        
    Regresa:
    --------
    subconjuntos: list(list(dict))
        Lista de M subconjuntos de datos.
    """
    n = len(datos)
    subconjuntos = []
    for _ in range(M):
        subconjunto = [random.choice(datos) for _ in range(n)]
        subconjuntos.append(subconjunto)
    return subconjuntos

def entrena_bosque(datos, target, clase_default, M, max_profundidad=None, acc_nodo=1.0, min_ejemplos=0, num_variables=None):
    """
    Entrena bosque aleatorio utilizando el criterio de entropía.
    
    Parámetros:
    -----------
    datos: list(dict)
        Lista de diccionarios, cada diccionario representa una instancia.
    target: str
       Atributo que se quiere predecir.
    clase_default: str
        Valor de la clase por default.
    M: int
        Número de árboles en el bosque.
    max_profundidad: int
        Maxima profundidad de los árboles. None = no hay límite.
    acc_nodo: int
        Porcentaje de acierto mínimo para considerar un nodo como hoja.
    min_ejemplos: int
        El número mínimo de ejemplos para considerar un nodo como hoja.
    num_variables: int
        El número de variables a considerar en cada nodo. Si es None, se consideran todas las variables.
        
    Regresa:
    --------
    bosque: list(NodoN)
        Lista de nodos raíz de los árboles entrenados.
    """
    subconjuntos = crea_subconjuntos(datos, M)
    bosque = []
    for subconjunto in subconjuntos:
        arbol = entrena_arbol(subconjunto, target, clase_default, max_profundidad, acc_nodo, min_ejemplos, num_variables)
        bosque.append(arbol)
    return bosque

def predice_bosque(bosque, datos):
    """
    Realiza predicciones utilizando un bosque aleatorio.
    
    Parámetros:
    -----------
    bosque: list(NodoN)
        Una lista de nodos raíz de los árboles entrenados.
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia.
        
    Regresa:
    --------
    predicciones: list
        Una lista de predicciones para cada instancia en los datos.
    """
    predicciones = []
    for instancia in datos:
        votos = [arbol.predice(instancia) for arbol in bosque]
        prediccion = max(set(votos), key=votos.count)
        predicciones.append(prediccion)
    return predicciones

def evalua_bosque(bosque, datos, target):
    """
    Evalúa la precisión de un bosque aleatorio.
    
    Parámetros:
    -----------
    bosque: list(NodoN)
        Una lista de nodos raíz de los árboles entrenados.
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia.
    target: str
        El nombre del atributo que se quiere predecir.
        
    Regresa:
    --------
    precision: float
        La precisión del bosque en los datos.
    """
    predicciones = predice_bosque(bosque, datos)
    return sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)