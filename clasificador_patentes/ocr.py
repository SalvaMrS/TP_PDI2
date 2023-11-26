import os
import string
import cv2
import numpy as np
import tensorflow as tf

class PatenteOCR:
    def __init__(self, ocr_model_path='clasificador_patentes/model/m4_1.1M_CPU', confianza_avg=0.3,
                 none_low_thresh=0.05):
        if not os.path.exists(ocr_model_path):
            raise FileNotFoundError(f'Modelo no encontrado en la ruta: {ocr_model_path}')

        self.imported = tf.saved_model.load(ocr_model_path)
        self.cnn_ocr_model = self.imported.signatures["serving_default"]
        self.alphabet = string.digits + string.ascii_uppercase + '_'
        self.confianza_avg = confianza_avg
        self.none_low_thresh = none_low_thresh

    def procesar_carpeta(self, carpeta):
        # Obtener la lista de archivos en la carpeta
        lista_archivos = os.listdir(carpeta)

        # Iterar sobre cada archivo en la carpeta
        for archivo in lista_archivos:
            # Comprobar si el archivo es una imagen
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construir la ruta completa del archivo
                ruta_completa = os.path.join(carpeta, archivo)

                # Leer la imagen
                img = cv2.imread(ruta_completa)

                # Verificar si la lectura de la imagen fue exitosa
                if img is not None:
                    # Llamar al método predict para procesar la imagen
                    patentes = self.predict_from_image(img)

                    # Hacer algo con las patentes detectadas, por ejemplo, imprimir en la consola
                    print(f"Patentes en {archivo}: {patentes}")

    def predict_from_image(self, frame: np.ndarray) -> list:
        # Supongamos que tienes una forma de obtener las coordenadas de las placas (iter_coords)
        iter_coords = obtener_coordenadas_de_placas(frame)

        patentes = self.predict(iter_coords, frame)
        return patentes

    def predict(self, iter_coords, frame: np.ndarray) -> list:
        patentes = []
        for yolo_prediction in iter_coords:
            x1, y1, x2, y2, _ = yolo_prediction
            plate, probs = self.predict_ocr(x1=x1, y1=y1, x2=x2, y2=y2, frame=frame)
            avg = np.mean(probs)
            # print(plate)
            if avg > self.confianza_avg and self.none_low(probs, thresh=self.none_low_thresh):
                plate = ''.join(plate).replace('_', '')
                patentes.append(plate)
        return patentes

    def none_low(self, probs, thresh=None):
        thresh = thresh or 0.3
        return all(prob >= thresh for prob in probs)

    def predict_ocr(self, *, x1, y1, x2, y2, frame: np.ndarray):
        cropped_plate = frame[y1:y2, x1:x2]
        prediction_ocr = self.__predict_from_array(cropped_plate)
        plate, probs = self.__probs_to_plate(prediction_ocr)
        return plate, probs

    def __probs_to_plate(self, prediction):
        prediction = prediction.reshape((7, 37))
        probs = np.max(prediction, axis=-1)
        prediction = np.argmax(prediction, axis=-1)
        plate = [self.alphabet[x] for x in prediction]
        return plate, probs

    def __predict_from_array(self, patente_recortada: np.ndarray):
        patente_recortada = cv2.cvtColor(patente_recortada, cv2.COLOR_RGB2GRAY)
        patente_recortada = cv2.resize(patente_recortada, (140, 70))
        patente_recortada = patente_recortada[np.newaxis, ..., np.newaxis]
        patente_recortada = tf.constant(patente_recortada, dtype=tf.float32) / 255.
        pred = self.cnn_ocr_model(patente_recortada)
        return pred[next(iter(pred))].numpy()
