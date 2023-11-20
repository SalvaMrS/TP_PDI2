import os
import cv2


image_conditions = {
    'img01.png': (800, None, 2.4),
    'img02.png': (12000, None, 1),
    'img03.png': (4000, 7000, 1),
    'img04.png': (4000, None, 1),
    'img05.png': (3000, 4000, 1),
    'img06.png': (7000, None, 2.4),
    'img07.png': (1000, 3000, 1),
    'img08.png': (2000, 2750, 1),
    'img09.png': (2000, 6000, 1),
    'img10.png': (3000, 3500, 1),
    'img11.png': (5000, 11000, 2.4),
    'img12.png': (800, 4000, 1)
}


class Detector:
    def __init__(self):
        pass

    def _process_image(self, imagen_path, area_min, area_max=None, aspect_ratio_min=1):
        image = cv2.imread(imagen_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.blur(gray, (3, 3))
        canny = cv2.Canny(gray_blur, 200, 200)
        canny = cv2.dilate(canny, None, iterations=3)
        cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)

            x, y, w, h = cv2.boundingRect(c)
            epsilon = 0.09 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            print(area)

            if area_min < area < (area_max if area_max is not None else float('inf')):
                aspect_ratio = float(w) / h
                if aspect_ratio > aspect_ratio_min:
                    # Crear la ruta para guardar la placa
                    placa_filename = imagen_path.split('/')[-1]
                    placa_filepath = os.path.join('Procesadas/', placa_filename)
                    placa = gray[y:y + h, x:x + w]
                    if not os.path.exists(placa_filepath):
                        cv2.imwrite(placa_filepath, placa)
                    cv2.imshow('PLACA', placa)
                    cv2.moveWindow('PLACA', 780, 10)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('Image', image)
        cv2.moveWindow('Image', 45, 10)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        for image_name, conditions in image_conditions.items():
            imagen_path = f'Patentes/Patentes/{image_name}'
            self._process_image(imagen_path, *conditions)




