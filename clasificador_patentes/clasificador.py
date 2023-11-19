from tensorflow.python.saved_model import tag_constants
from clasificador_patentes.ocr import PatenteOCR
import numpy as np
import cv2
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
class DetectorPatente:
    MODEL_PATH = 'clasificador_patentes/model/tf-yolo_tiny_v4-512x512-custom-anchors'
    INPUT_SIZE = 512
    IOU_THRESHOLD = 0.45
    SCORE_THRESHOLD = 0.25

    def __init__(self):
        self.saved_model_loaded = tf.saved_model.load(self.MODEL_PATH, tags=[tag_constants.SERVING])
        self.yolo_infer = self.saved_model_loaded.signatures['serving_default']

    def preprocess(self, frame: np.ndarray) -> tf.Tensor:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralado
        _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Aplicar operaciones morfológicas para mejorar la detección de contornos
        kernel = np.ones((5, 5), np.uint8)
        morphological = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Dilatación adicional para resaltar los contornos
        morphological = cv2.dilate(morphological, kernel, iterations=1)

        # Encuentra los contornos en la imagen
        contours, _ = cv2.findContours(morphological, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si se encontraron contornos, dibuja un rectángulo alrededor del área del contorno más grande (asumiendo que es la patente)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Redimensionar la imagen a las dimensiones de entrada del modelo
        resized_frame = cv2.resize(frame, (512, 512))

        # Mostrar la imagen con el rectángulo dibujado
        cv2.imshow('Imagen con rectángulo de patente', resized_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Normalizar la imagen restando la media y dividiendo por la desviación estándar
        mean = np.mean(resized_frame)
        std = np.std(resized_frame)
        normalized_frame = (resized_frame - mean) / std

        # Añadir una dimensión para representar el batch (1 imagen)
        normalized_frame = np.expand_dims(normalized_frame, axis=0)

        # Convertir a tf.Tensor y asegurarse de que sea float32
        return normalized_frame.astype(np.float32)

    def correct_perspective(self, frame: np.ndarray) -> np.ndarray:
        # Implementa tu lógica de corrección de perspectiva aquí
        # Puedes usar cv2.warpPerspective para esto
        # Ejemplo simple:
        pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        corrected_frame = cv2.warpPerspective(frame, matrix, (300,300))
        return corrected_frame

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        # Implementa tu lógica de umbralización adaptativa aquí
        # Puedes usar cv2.adaptiveThreshold para esto
        # Ejemplo simple:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def predict(self, input_img: tf.Tensor) -> dict:
        print(input_img)
        return self.yolo_infer(input_img)

    def procesar_salida_yolo(self, output: dict) -> list:
        for key, value in output.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.IOU_THRESHOLD,
            score_threshold=self.SCORE_THRESHOLD
        )
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    def draw_bboxes(self, frame: np.ndarray, bboxes: list, mostrar_score: bool = False) -> np.ndarray:
        for x1, y1, x2, y2, score in self.yield_coords(frame, bboxes):
            font_scale = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            if mostrar_score:
                cv2.putText(frame, f'{score:.2f}%', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (20, 10, 220), 5)
        return frame

    def show_predicts(self, frame: np.ndarray):
        input_img = self.preprocess(frame)
        print("Input shape:", input_img.shape)
        yolo_out = self.predict(input_img)
        """
        bboxes = self.procesar_salida_yolo(yolo_out)
        # Convertir el frame completo a escala de grises una vez

        iter_coords = self.yield_coords(frame, bboxes)

        for x1, y1, x2, y2, _ in iter_coords:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate, probs = PatenteOCR.predict_ocr(x1, y1, x2, y2, frame)
            # print(plate)
            avg = np.mean(probs)

            if avg > self.ocr.confianza_avg:
                plate = ''.join(plate).replace('_', '')
                print(plate)
                return plate
        """

    @staticmethod
    def yield_coords(frame: np.ndarray, bboxes: list):
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        image_h, image_w, _ = frame.shape
        for i in range(num_boxes[0]):
            coor = out_boxes[0][i]
            x1 = int(coor[1] * image_w)
            y1 = int(coor[0] * image_h)
            x2 = int(coor[3] * image_w)
            y2 = int(coor[2] * image_h)
            yield x1, y1, x2, y2, out_scores[0][i]
