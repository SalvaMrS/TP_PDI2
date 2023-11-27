import cv2
import os
import numpy as np

class SegmentadorDeImagenes:
    def __init__(self, segment_width=80, segment_height=100):
        self.segment_width = segment_width
        self.segment_height = segment_height

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
                    # Llamar al método segmentar_y_guardar para procesar la imagen
                    self.segmentar_y_guardar(img)
    @staticmethod
    def segmentar_y_guardar(subimg):
        subimg_gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        _, subimg_humbr = cv2.threshold(subimg_gray, 90, 255, cv2.THRESH_OTSU)
        tmp = _ + (_ * 0.45)
        _, subimg_humbr = cv2.threshold(subimg_gray, tmp, 255, cv2.THRESH_BINARY)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(subimg_humbr)

        result_image = subimg.copy()

        for i in range(1, retval):  # Comienza desde 1 para omitir el componente de fondo (etiqueta 0)
            obj = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            area = cv2.contourArea(contours[0])

            if area < 10 or area > 160:
                continue

            epsilon = 0.04 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)

            if len(approx) > 3:
                cv2.drawContours(result_image, [approx], -1, (255, 0, 0), 1)
                st = stats[i, :]
                cv2.rectangle(result_image, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 0, 255),
                              thickness=1)

        result_image = cv2.resize(result_image, (600, 200))
        cv2.imshow('Objetos Filtrados', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
