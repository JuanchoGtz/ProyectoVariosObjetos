
from flask import Flask,render_template,request,Response

import tensorflow
from tensorflow.keras.preprocessing import image
import cv2

from werkzeug.utils import secure_filename
import os

import numpy

#Cargar previamente el modelo cargado 
modelo= tensorflow.keras.models.load_model("modelo_clasificador_objetos.h5")

clases=["adidas","ceraza","gato","guitarra","kiwi","leon","nike","perro","tigre","violin"]

servidor=Flask(__name__)

UPLOAD_FOLDER="static/uploads"

servidor.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@servidor.route("/")
def index():
    return render_template("index.html")

@servidor.route("/prediccion", methods=["POST"])
def prediccion():

    imagen=request.files["imagen"]

    ruta_imagen=os.path.join(servidor.config["UPLOAD_FOLDER"], secure_filename(imagen.filename))
    imagen.save(ruta_imagen)

    imagen=image.load_img(f"/Users/Juanchopc/Documents/UBAM/11vo/IA/ProyectoVariosObjetos/{ruta_imagen}", target_size=(128,128))
    
    img_array=image.img_to_array(imagen)

    img_array=numpy.expand_dims(img_array,axis=0)

    img_array=img_array/255.0

    print("RUTA IMAGEN", f"/Users/Juanchopc/Documents/UBAM/11vo/IA/ProyectoVariosObjetos/{ruta_imagen}")

    prediccion=modelo.predict(img_array)

    print("PREDICCION",prediccion)

    clase_predicha=clases[numpy.argmax(prediccion)]

    return render_template("index.html", clase_predicha=clase_predicha, ruta_imagen= ruta_imagen)

if __name__=="__main__":
    servidor.run(port=4000, debug=True)

#añadir lo de la camara