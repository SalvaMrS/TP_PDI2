import cv2
from clasificador.Funciones import obtener_contornos, monedas_y_dados

ruta_imagen = './clasificador/imagenes/monedas.jpg'

img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

img_bin = obtener_contornos(img)

cv2.imwrite('./clasificador/imagenes/imagen_contornos.jpg', img_bin)