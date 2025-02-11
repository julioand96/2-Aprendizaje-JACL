import utileria as ut
import bosque_aleatorio as ba
import random
import pandas as pd

# Cargar los datos del archivo wine.data
data_path = "wine/wine.data"
column_names = [
    'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]
data = pd.read_csv(data_path, header=None, names=column_names)

# Convertir a lista de diccionarios
datos = data.to_dict(orient='records')

# Selecciona los atributos
target = 'Class'
atributos = column_names

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Parámetros del bosque
M = 5  # Número de árboles en el bosque
num_variables = 5  # Número de variables a considerar en cada nodo

# Entrena el bosque
bosque = ba.entrena_bosque(
    datos_entrenamiento,
    target,
    clase_default=1,
    M=M,
    max_profundidad=5,
    num_variables=num_variables
)

# Evalúa el bosque
error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)

# Muestra los errores
print('Bosque Aleatorio'.center(30))
print('-' * 30)
print('Conjunto'.center(15) + 'Error'.center(15))
print('-' * 30)
print('Entrenamiento'.center(15) + f'{error_en_muestra:.2f}'.center(15))
print('Validación'.center(15) + f'{error_en_validacion:.2f}'.center(15))
print('-' * 30 + '\n')

# Entrena con todos los datos
bosque_completo = ba.entrena_bosque(
    datos,
    target,
    clase_default=1,
    M=M,
    max_profundidad=5,
    num_variables=num_variables
)
error_completo = ba.evalua_bosque(bosque_completo, datos, target)
print(f'Error del modelo seleccionado entrenado con TODOS los datos: {error_completo:.2f}')