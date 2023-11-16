"""import cv2
import numpy as np

class DetectorObjetos:
    def __init__(self, imagen_path):
        self.imagen = cv2.imread(imagen_path)
        self.gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        self.umbral = cv2.threshold(self.gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.contornos, _ = cv2.findContours(self.umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def segmentar_objetos(self, area_minima=5):
        contornos_filtrados = [contorno for contorno in self.contornos if cv2.contourArea(contorno) > area_minima]
        cv2.drawContours(self.imagen, contornos_filtrados, -1, (0, 255, 0), 2)

    def mostrar_segmentacion(self):
        ancho_max = 800
        altura, ancho, _ = self.imagen.shape
        nuevo_ancho = ancho_max
        nuevo_alto = int((ancho_max / ancho) * altura)

        imagen_redimensionada = cv2.resize(self.imagen, (nuevo_ancho, nuevo_alto))
        #imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Segmentación', imagen_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Crear un objeto DetectorObjetos
detector = DetectorObjetos('monedas.jpg')


# Segmentar monedas con un área mínima de 500
detector.segmentar_objetos()

# Mostrar la segmentación
detector.mostrar_segmentacion()
"""

import cv2
import numpy as np

class DetectorObjetos:
    def __init__(self, imagen_path):
        self.imagen = self.redimensionar_imagen(imagen_path, 800, 600)
        self.gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        self.histograma_equilibrado = self.ecualizar_histograma(self.gris)
        self.umbral = cv2.threshold(self.histograma_equilibrado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.contornos, _ = cv2.findContours(self.umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def ecualizar_histograma(self, imagen):
        equ = cv2.equalizeHist(imagen)
        return equ

    def redimensionar_imagen(self, imagen_path, nuevo_ancho, nuevo_alto):
        imagen = cv2.imread(imagen_path)
        return cv2.resize(imagen, (nuevo_ancho, nuevo_alto))

    def segmentar_objetos(self, area_minima=5):
        contornos_filtrados = [contorno for contorno in self.contornos if cv2.contourArea(contorno) > area_minima]
        cv2.drawContours(self.imagen, contornos_filtrados, -1, (0, 255, 0), 2)

    def mostrar_histograma(self, imagen, titulo='Histograma'):
        histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        cv2.imshow(titulo, histograma)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mostrar_segmentacion(self):
        cv2.imshow('Segmentación', self.imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Crear un objeto DetectorObjetos
detector = DetectorObjetos('monedas.jpg')

# Mostrar el histograma antes de la segmentación
detector.mostrar_histograma(detector.gris, 'Histograma Antes')

# Segmentar monedas con un área mínima de 5
detector.segmentar_objetos()

# Mostrar el histograma después de la segmentación
detector.mostrar_histograma(detector.umbral, 'Histograma Después')

# Mostrar la segmentación
detector.mostrar_segmentacion()
