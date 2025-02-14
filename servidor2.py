from flask import Flask, render_template, request, Response, jsonify

import tensorflow
from tensorflow.keras.preprocessing import image
import cv2

from werkzeug.utils import secure_filename
import os

import numpy

modelo = tensorflow.keras.models.load_model("modelo_clasificador_frutas.h5")

#cambiar a las clases del proyecto
clases = ["Mango", "Manzana", "Naranja", "Platano", "Uva"]

servidor = Flask(__name__)

camara = cv2.VideoCapture(0)

prediccion_mas_reciente = ""

UPLOAD_FOLDER = "static/uploads"
servidor.config[UPLOAD_FOLDER] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@servidor.route("/")
def index():
    return render_template("index3.html")

@servidor.route('/prediccion_reciente')
def prediccion_reciente():
    return jsonify({'prediccion_mas_reciente': prediccion_mas_reciente})

def generar_frames():
    global prediccion_mas_reciente

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        frame_redimensionado =cv2.resize(frame, 128, 128)
        frame_normalizado = frame_redimensionado / 255.0
        frame_expandido = numpy.expand_dims(frame_normalizado, axis=0)

        prediccion = modelo.predict(frame_expandido)
        clase_predicha = clases[numpy.argmax(prediccion)]
        prediccion_mas_reciente = clase_predicha

        _, buffer = cv2.imencode('.jpg', frame)
        frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')
        
@servidor.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    servidor.run(port=4000, debug=True)