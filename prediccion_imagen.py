import tensorflow
import numpy
from tensorflow.keras.preprocessing import image

        # Cargar el modelo previamente guardado
modelo =tensorflow.keras.models.load_model ("modelo_clasificador_fresas_moras.h5")
        # Ruta de la imagen que quieres clasificar
ruta_imagen = "/Users/Juanchopc/Documents/UBAM/11vo/IA/PracticaMoras/mora.jpg"
        # Cargar la imagen con el tamaño esperado por la red (128x128)py pre
imagen = image.load_img (ruta_imagen, target_size= (128, 128))
        # Convertir la imagen a un array de numpy
img_array = image.img_to_array (imagen)
        # Expandir dimensiones para que coincida con el formato esperado por elmodelo (batch_size, alto, ancho, canales)
img_array = numpy. expand_dims (img_array, axis=0)
        # Normalizar los valores de los pixeles (como se hizo en el entrenamiento)
img_array = img_array / 255.0
        # Hacer la predicción
prediccion = modelo. predict (img_array)
        # Interpretar el resultado
        # Los valores se asignan dependiendo el orden alfabetico de las carpetas
        # Como m de manzana es primero, se le asigna 0
        # Como p de platano es después, se le asigna 1
        # Menor de 0.5 es manzana
if prediccion [0] [0] <= 0.5:
    print (f"Predicción: (prediccion), Es una Fresa")
        # Mayor de 0.5 es platano
else:
    print(f" Predicción: (prediccion). Es una Mora")

