

import cv2
import tensorflow as tf
import numpy as np 

#Cargar el modelo entremado que se  guardo en un archivo
modelo = tf.keras.models.load_model("modelo_clasificador_fresas_moras.h5")

#Configuracion de parametros

#Tamaño al que redimensionar las imagenes
ancho_imagen, alto_imagen = 128,128

#etiquetas de clases
clases={0: "Fresa", 1:"Moras"}

#Inicializa la camara web

#Usa la camara por defecto 
camara=cv2.VideoCapture(0)

if not camara.isOpened():
    print ("Error al acceder a la camara")
    exit()

print ("Presiona 'q' para Salir")

while True:
    #capturar un frame de la camara 
    ret, frame =camara.read()
    if not ret:
        print("No se pudo capturar la imagen saliendo....")
        break
    #Mostrar frame en tiempo real 
    cv2.imshow("Camara en Vivo",frame)

    #preprocesar la imagen capturada
    #Redimensionar la imagen con las medidas que espera la red convolucional
    imagen_procesada=cv2.resize(frame,(ancho_imagen,alto_imagen))

    #normaliza la imagen es decir pasarlos a escalas grises
    imagen_procesada= imagen_procesada/255.0
    imagen_procesada= np.expand_dims(imagen_procesada, axis=0) 

    prediccion=modelo.predict(imagen_procesada)

    etiqueta= None

    if(prediccion[0][0]<=0.5):
        etiqueta="Fresa"

    else:
        etiqueta="Mora"

    cv2.putText(frame, f"Prediccion: {prediccion[0][0]:.2f} - {etiqueta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("Prediccion", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()