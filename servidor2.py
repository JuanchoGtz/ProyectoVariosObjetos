from flask import Flask, render_template, Response, jsonify
import tensorflow as tf
import cv2
import numpy as np

# Cargar modelo
modelo = tf.keras.models.load_model("/Users/Juanchopc/Documents/UBAM/11vo/IA/ProyectoVariosObjetos/modelo_clasificador_objetos.h5")

# Clases de objetos
clases = ["adidas", "cereza", "gato", "guitarra", "kiwi", "leon", "nike", "perro", "tigre", "violin"]

# Iniciar Flask
servidor = Flask(__name__)

# Configurar cámara para frame más pequeño (160x120)
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# Variable global para la predicción
prediccion_mas_reciente = ""

@servidor.route("/")
def index():
    return render_template("index2.html")

@servidor.route('/prediccion_reciente')
def prediccion_reciente():
    return jsonify({'prediccion_mas_reciente': prediccion_mas_reciente})

def generar_frames():
    global prediccion_mas_reciente

    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        # Redimensionar frame para predicción sin afectar la calidad del video mostrado
        frame_redimensionado = cv2.resize(frame, (128, 128))
        frame_normalizado = frame_redimensionado / 255.0
        frame_expandido = np.expand_dims(frame_normalizado, axis=0)

        # Realizar predicción
        prediccion = modelo.predict(frame_expandido)
        clase_predicha = clases[np.argmax(prediccion)]
        prediccion_mas_reciente = clase_predicha

        # Convertir frame a formato JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')

@servidor.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    servidor.run(port=4000, debug=True)
