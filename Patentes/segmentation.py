import cv2
import numpy as np

class SegmentationProcessor:
    def __init__(self, image_matrix):
        self.image = image_matrix
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.threshold = None
        self.contours = None
        self.hough_lines = None

    def preprocess_image(self):
        _, self.threshold = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def find_contours(self):
        contours, _ = cv2.findContours(self.threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def apply_hough_transform(self):
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        self.hough_lines = lines

    def draw_hough_lines(self):
        hough_lines_image = self.image.copy()
        if self.hough_lines is not None:
            for line in self.hough_lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return hough_lines_image

    def get_letter_boxes(self):
        letter_boxes = []
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            letter_boxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2) - coordenadas de la esquina superior izquierda y la esquina inferior derecha
        return letter_boxes

    def draw_letter_boxes(self):
        segmented_image = self.image.copy()
        for box in self.get_letter_boxes():
            x1, y1, x2, y2 = box
            cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return segmented_image

    def process_image(self):
        self.preprocess_image()
        self.find_contours()
        self.apply_hough_transform()
        result_image = self.draw_hough_lines()
        return result_image
