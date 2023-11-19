from tensorflow.python.saved_model import tag_constants
from clasificador_patentes.ocr import PatenteOCR
import numpy as np
import cv2
import tensorflow as tf

class DetectorPatente:
    MODEL_PATH = 'clasificador_patentes/model/tf-yolo_tiny_v4-512x512-custom-anchors'
    INPUT_SIZE = 512
    IOU_THRESHOLD = 0.45
    SCORE_THRESHOLD = 0.1

    def __init__(self):
        self.saved_model_loaded = tf.saved_model.load(self.MODEL_PATH, tags=[tag_constants.SERVING])
        self.yolo_infer = self.saved_model_loaded.signatures['serving_default']

    def preprocess(self, frame: np.ndarray) -> tf.Tensor:
        corrected_frame = self.correct_perspective(frame)

        # Asegúrate de que la imagen tenga el rango correcto (0-1)
        image_data = corrected_frame.astype(np.float32) / 255.0
        # Resize de la imagen a las dimensiones de entrada del modelo
        image_data = cv2.resize(image_data, (512, 512))
        # Aumento de datos aleatorio: ajuste de brillo y contraste
        alpha = 0.5 + np.random.uniform(-0.2, 0.2)
        beta = 0.5 + np.random.uniform(-0.2, 0.2)
        image_data = cv2.convertScaleAbs(image_data, alpha=alpha, beta=beta)

        # Aumento de datos aleatorio: rotación
        angle = np.random.uniform(-10, 10)
        rows, cols, _ = image_data.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image_data = cv2.warpAffine(image_data, rotation_matrix, (cols, rows))



        # Normaliza la imagen canal por canal restando la media y dividiendo por la desviación estándar
        mean = np.array([0.485, 0.456, 0.406])  # Media de ImageNet
        std = np.array([0.229, 0.224, 0.225])  # Desviación estándar de ImageNet
        image_data = (image_data - mean) / std

        # Añade una dimensión para representar el batch (1 imagen)
        image_data = np.expand_dims(image_data, axis=0)

        # Convierte a tf.Tensor y asegúrate de que sea float32
        return tf.constant(image_data, dtype=tf.float32)

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
        # cv2.imwrite('test.jpg', frame)
        input_img = self.preprocess(frame)
        yolo_out = self.predict(input_img)
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
