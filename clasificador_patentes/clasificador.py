from tensorflow.python.saved_model import tag_constants
import numpy as np
import cv2
import tensorflow as tf

class DetectorPatente:
    """
    Module responsible for license plate detection.
    """

    MODEL_PATH = 'clasificador_patentes/model/tf-yolo_tiny_v4-512x512-custom-anchors'
    INPUT_SIZE = 512
    IOU_THRESHOLD = 0.45
    SCORE_THRESHOLD = 0.25

    def __init__(self):
        """
        Initialize the PlateDetector with the fixed YOLO model.
        """
        self.saved_model_loaded = tf.saved_model.load(self.MODEL_PATH, tags=[tag_constants.SERVING])
        self.yolo_infer = self.saved_model_loaded.signatures['serving_default']

    def preprocess(self, frame: np.ndarray) -> tf.Tensor:
        """
        Mejora el preprocesamiento de la imagen.

        Parameters:
            frame (np.ndarray): Imagen de entrada.

        Returns:
            tf.Tensor: Tensor de imagen preprocesado.
        """
        # Asegúrate de que la imagen tenga el rango correcto (0-1)
        image_data = frame.astype(np.float32) / 255.0

        # Aumento de datos aleatorio: ajuste de brillo y contraste
        alpha = 0.5 + np.random.uniform(-0.2, 0.2)  # factor de ajuste de contraste
        beta = 0.5 + np.random.uniform(-0.2, 0.2)   # factor de ajuste de brillo
        image_data = cv2.convertScaleAbs(image_data, alpha=alpha, beta=beta)

        # Aumento de datos aleatorio: rotación
        angle = np.random.uniform(-10, 10)  # ángulo de rotación en grados
        rows, cols, _ = image_data.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image_data = cv2.warpAffine(image_data, rotation_matrix, (cols, rows))

        # Resize de la imagen a las dimensiones de entrada del modelo
        image_data = cv2.resize(image_data, (512, 512))

        # Normaliza la imagen restando la media y dividiendo por la desviación estándar
        mean = [0.485, 0.456, 0.406]  # Media de ImageNet
        std = [0.229, 0.224, 0.225]   # Desviación estándar de ImageNet
        image_data = (image_data - mean) / std

        # Añade una dimensión para representar el batch (1 imagen)
        image_data = np.expand_dims(image_data, axis=0)

        # Convierte a tf.Tensor
        return tf.constant(image_data)

    def predict(self, input_img: tf.Tensor) -> dict:
        """
        Make an inference based on the input tensor.

        Parameters:
            input_img (tf.Tensor): Input image tensor.

        Returns:
            dict: Output of the YOLO model.
        """
        return self.yolo_infer(input_img)

    def procesar_salida_yolo(self, output: dict) -> list:
        """
        Process the output of YOLO.

        Parameters:
            output (dict): Output dictionary from YOLO.

        Returns:
            list: Bounding Boxes of all detected license plates after NMS.
        """
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
        """
        Draw bounding boxes on the frame.

        Parameters:
            frame (np.ndarray): Original frame.
            bboxes (list): Predictions after NMS.
            mostrar_score (bool): Whether to display scores.

        Returns:
            np.ndarray: Frame with drawn bounding boxes.
        """
        for x1, y1, x2, y2, score in self.yield_coords(frame, bboxes):
            font_scale = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(frame, f'{score:.2f}%', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (20, 10, 220), 5)
        return frame

    @staticmethod
    def yield_coords(frame: np.ndarray, bboxes: list):
        """
        Yield coordinates of the rectangles.

        Parameters:
            frame (np.ndarray): Original frame.
            bboxes (list): Predictions after NMS.

        Yields:
            tuple: Coordinates and score.
        """
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        image_h, image_w, _ = frame.shape
        for i in range(num_boxes[0]):
            coor = out_boxes[0][i]
            x1 = int(coor[1] * image_w)
            y1 = int(coor[0] * image_h)
            x2 = int(coor[3] * image_w)
            y2 = int(coor[2] * image_h)
            yield x1, y1, x2, y2, out_scores[0][i]