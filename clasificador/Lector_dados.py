import cv2
import numpy as np

from clasificador.Funciones import monedas_y_dados, es_circulo, imagen_dado, cant_puntos, \
    dibujar_dado, imshow

img = cv2.imread('./clasificador/imagenes/imagen_contornos.jpg', cv2.IMREAD_GRAYSCALE)

_, dados = monedas_y_dados(img)

# Cargar la imagen
imagen_resultado = cv2.imread('./clasificador/imagenes/monedas_resultado.jpg')
# Convertir la imagen a formato RGB
imagen_resultado = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)

for contorno in dados:
    imagen_original = cv2.imread('./clasificador/imagenes/monedas.jpg', cv2.IMREAD_GRAYSCALE)

    # Mostrar la región de interés recortada
    imagen = imagen_dado(imagen_original, contorno)

    puntos = cant_puntos(imagen)

    # Supongamos que 'contorno' está definido y 'color' es un tuple (B, G, R)
    imagen_resultado = dibujar_dado(imagen_resultado, contorno, (0, 0, 0), puntos)

# Guardar la imagen resultante
cv2.imwrite('./clasificador/imagenes/monedas_resultado.jpg', cv2.cvtColor(imagen_resultado, cv2.COLOR_RGB2BGR))