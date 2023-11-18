import cv2
from clasificador.Funciones import obtener_contornos, monedas_y_dados

ruta_imagen_entrada = './clasificador/imagenes/monedas.jpg'
imagen_gris = cv2.imread(ruta_imagen_entrada, cv2.IMREAD_GRAYSCALE)

imagen_contornos = obtener_contornos(imagen_gris)

ruta_imagen_salida = './clasificador/imagenes/imagen_contornos.jpg'
cv2.imwrite(ruta_imagen_salida, imagen_contornos)