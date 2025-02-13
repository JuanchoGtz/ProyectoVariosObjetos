

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard


# Configuración de parámetros
# Ancho y alto de las imágenes
ancho_imagen, alto_imagen = 128, 128  

# Número de imagenes que la red toma a la vez para el entrenamiento y validación
tamano_lote = 32  

# Número de veces que la red analiza el conjunto de imagenes para entrenar
# A mayor número de epocas que la red entrene, obtendrá mejores resultados
epocas = 500



# =======    1

# Carpetas donde se encuentran las imagenes de entrenamiento y validación
directorio_entrenamiento = "/Users/Juanchopc/Documents/UBAM/11vo/IA/ProyectoVariosObjetos/dataset/entrenamiento"

directorio_validacion = "/Users/Juanchopc/Documents/UBAM/11vo/IA/ProyectoVariosObjetos/dataset/validacion" 


# El siguiente bloque de codigo, toma las imagenes de entrenamiento
# y les simula distorsiones de distintos tipos
# Para reconocer los objetos desde diferentes posiciones, distancias, movimientos

generador_entrenamiento = ImageDataGenerator(
    
    rescale = 1.0 / 255,
    rotation_range = 50,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    vertical_flip = True,
    horizontal_flip = True,
)


generador_validacion = ImageDataGenerator(rescale=1.0 / 255)

# Carga de datos desde las carpetas
datos_entrenamiento = generador_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",
)

datos_validacion = generador_validacion.flow_from_directory(
    directorio_validacion,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",
)

# Definición de la ESTRUCTURA DE LA RED neuronal convolucional
# Aquí se crean las capas necesarias para la red
# Se puede probar con más o con menos capas para evaluar el resultado obtenido
RED_NEURONAL_CONVOLUCIONAL = models.Sequential([

    # =======    2, 3


    # Primero una capa convolucional que usará 32 filtros que miden 3x3
    # Tendrá una función de activación RELU
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(ancho_imagen, alto_imagen, 3)),

    # =======    4

    # Luego una capa de reducción
    layers.MaxPooling2D((2, 2)),

    # Luego otra capa convolucional
    # Esta tendrá 64 filtros (para extraer características mas abstractas) que miden 3x3
    layers.Conv2D(64, (3, 3), activation="relu"),

    # Otra capa de reducción
    layers.MaxPooling2D((2, 2)),

    # Otra capa de convolución ahora con 128 filtros
    # Entre más filtros más características del objeto detecta la capa
    layers.Conv2D(128, (3, 3), activation="relu"),

    # Otra capa de reducción
    layers.MaxPooling2D((2, 2)),

    # Se utiliza la función Dropout para apagar aleatoriamente la mitad de las neuronas
    # Para que se activen unas y luego otras
    # Y evitar que haya neuronas muertas en la red
    layers.Dropout(0.5),
    
    # =======    5

    # ============= Se crea después una red neuronal "tipica" ==================

    # Que es la que va clasificar la imagen de acuerdo a todas las caracteristicas 
    # Que las capas convolucionales identificaron

    # Primero se "aplanan" los datos obtenidos y después se introducen a la red
    layers.Flatten(),

    # Esta red tiene solo 2 capas
    # Una con 128 neuronas que empezará a clasificar la imagen
    layers.Dense(128, activation="relu"),

    # Una capa de salida con una sola neurona con activacion sigmoide (valores entre 0 y 1)
    # Donde valores cercanos a cero serán una fruta y valores cercanos a 1 serán otra fruta
    layers.Dense(10, activation="softmax"), 
])



# Compilación de la red
RED_NEURONAL_CONVOLUCIONAL.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Esto nos permite guardar los resultados del entrenamiento
# Y visualizarlos en una herramienta web con graficas
#resultados_entrenamiento_red_convolucional = TensorBoard(log_dir="/Users/Juanchopc/Documents/UBAM/11vo/IA/PracticaNueva")


# Entrenamiento de la red
# En esta parte se toman los datos de entrenamiento
# Se le indica 
entrenar_red = RED_NEURONAL_CONVOLUCIONAL.fit(
    datos_entrenamiento,
    epochs=epocas,
    validation_data=datos_validacion,
    #callbacks = [resultados_entrenamiento_red_convolucional]
)


# Evaluación de la red

# Este código nos irá mostrando como va aprendiendo la red conforme avanza su entrenamiento
puntaje = RED_NEURONAL_CONVOLUCIONAL.evaluate(datos_validacion)
print(f"Pérdida: {puntaje[0]:.4f}, Precisión: {puntaje[1]:.4f}")


# Cuando la red termina de entrenar con los parametros y datos utilizados
# Arroja como resultado algo llamado MODELO

# Guardar el modelo en un archivo para usarlo después
RED_NEURONAL_CONVOLUCIONAL.save("modelo_clasificador_objetos.h5")

# Guardar igual en un archivo, los pesos que usó el modelo
RED_NEURONAL_CONVOLUCIONAL.save_weights("pesos.weights.h5")

