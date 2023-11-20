from tensorflow.python.saved_model import tag_constants
from ocr import PatenteOCR
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

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralado
        _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Aplicar operaciones morfológicas para mejorar la detección de contornos
        kernel = np.ones((5, 5), np.uint8)
        morphological = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Dilatación adicional para resaltar los contornos
        morphological = cv2.dilate(morphological, kernel, iterations=1)

        # Encuentra los contornos en la imagen original para dibujarlos
        contours, _ = cv2.findContours(morphological, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar contornos en la imagen original
        
        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Mostrar la imagen con contornos dibujados
        #cv2.imshow('Imagen con Contornos', frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return morphological

    def postprocess_contours(self, image, contours):
        # Inicializar lista para almacenar coordenadas de la patente
        patente_coords = []

        for contour in contours:
            area = cv2.contourArea(contour)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and area > 2000 and area < 5000:  # Filtro de área y forma
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                patente_coords.append(box)

        return image, patente_coords

    def apply_additional_filters(self, image):
        # Aplicar más filtros y operaciones morfológicas según sea necesario
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Aplicar operaciones morfológicas adicionales
        kernel = np.ones((5, 5), np.uint8)
        morphological = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        morphological = cv2.dilate(morphological, kernel, iterations=1)

        # Puedes experimentar con más filtros y operaciones aquí

        return morphological

    def perspective_transform(self, image, box):
        rect = np.array(box, dtype="float32")
        width, height = 512, 512  # Ajusta las dimensiones según sea necesario
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped

    def detect_and_segment(self, image_path):
        image = cv2.imread(image_path)
        processed_image = self.preprocess(image.copy())
        # Convertir la imagen procesada a escala de grises
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # Aplicar descenso de gradiente para resaltar bordes
        gradient_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_image = np.uint8(np.absolute(gradient_image))
        
        # Combinar imagen original con bordes resaltados
        processed_image = cv2.addWeighted(processed_image, 0.7, cv2.cvtColor(gradient_image, cv2.COLOR_GRAY2BGR), 0.3, 0)

         # Convertir la imagen procesada a escala de grises nuevamente
        gray_processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral para obtener una imagen binaria
        _, binary_image = cv2.threshold(gray_processed_image, 50, 255, cv2.THRESH_BINARY)

        # Detección de contornos
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Postprocesamiento de contornos aplicando más filtros y operaciones morfológicas
        result_image, patente_coords = self.postprocess_contours(image.copy(), contours)

        # Dibuja solo la patente en la imagen original
        for box in patente_coords:
            roi = self.perspective_transform(image.copy(), box)
            cv2.imshow('Segmento de Patente', roi)

        # Aplicar filtros adicionales y operaciones morfológicas
        additional_processed_image = self.apply_additional_filters(image.copy())

        cv2.imshow('Contornos Detectados (Postprocesamiento)', result_image)
        cv2.imshow('Imagen con Filtros Adicionales', additional_processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
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