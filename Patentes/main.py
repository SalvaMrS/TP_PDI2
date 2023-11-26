import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from Funciones import *
from Patentes.segmentation import SegmentationProcessor

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

carpeta_imagenes = "./Patentes/Imagenes/"
imagenes = cargar_imagenes(carpeta_imagenes)
imagenes_grises = transformar_imagenes(imagenes, cv2.cvtColor, {'code': cv2.COLOR_BGR2GRAY})
imagenes_sobel = transformar_imagenes(imagenes_grises, filtro_sobel)
imagenes_sobel_umbralizado = transformar_imagenes(imagenes_sobel, aplicar_umbral, {'umbral': 100})
kernel_dilatacion = np.ones((4, 4), np.uint8)
imagenes_sobel_dilatado = transformar_imagenes(imagenes_sobel_umbralizado, cv2.dilate, {'kernel':kernel_dilatacion})
kernel_erosion = np.ones((5, 1), np.uint8)
parametros_erosion = {'kernel': kernel_erosion, 'iterations': 2}
imagenes_sobel_erocion = transformar_imagenes(imagenes_sobel_dilatado, cv2.erode, parametros_erosion)
x = 16
y = 4
kernel_pasa_bajo = np.ones((x, y), np.float32) / (x * y)
parametros_filtro_pasa_bajo = {'ddepth': -1, 'kernel': kernel_pasa_bajo}
imagenes_sobel_pasa_bajo = transformar_imagenes(imagenes_sobel_erocion, cv2.filter2D, parametros_filtro_pasa_bajo)
umbral_binario = 220
imagenes_sobel_binario = transformar_imagenes(imagenes_sobel_pasa_bajo, aplicar_umbral, {'umbral': umbral_binario})
kernel_erosion2 = np.ones((1, 5), np.uint8)
parametros_erosion2 = {'kernel': kernel_erosion2, 'iterations': 2}
imagenes_sobel_erocion2 = transformar_imagenes(imagenes_sobel_binario, cv2.erode, parametros_erosion2)
kernel_eliptico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 22))
parametros_clausura = {'op': cv2.MORPH_CLOSE,'kernel': kernel_eliptico, 'iterations': 1}
imagenes_sobel_clausura = transformar_imagenes(imagenes_sobel_erocion2, cv2.morphologyEx, parametros_clausura)
umbral = 200
imagenes_sobel_filtrada = transformar_imagenes(imagenes_sobel_clausura, eliminar_objetos_pequenos, {'umbral_tamano_minimo': umbral})
imagenes_sobel_mancha_mas_abajo = transformar_imagenes(imagenes_sobel_filtrada, mantener_objeto_mas_abajo)
kernel_dilatacion = np.ones((20, 30), np.uint8)
imagenes_sobel_umbralizado2 = transformar_imagenes(imagenes_sobel_mancha_mas_abajo, cv2.dilate, {'kernel':kernel_dilatacion})
tuplas_imagenes = list(zip(imagenes, imagenes_sobel_umbralizado2))
subimagenes_zona_seleccionada = transformar_imagenes(tuplas_imagenes, obtener_subimagen_zona_seleccionada)
for subimg in subimagenes_zona_seleccionada:
    gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    _, saturated_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 100, 170, apertureSize=3)


    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=50)

    if lines is not None:
        # Encontrar la línea dominante (la línea más larga)
        longest_line = max(lines, key=lambda x: x[0][0] + x[0][1])
        rho, theta = longest_line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Calcular el ángulo de la línea
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        # Calcular la matriz de transformación para enderezar la imagen
        rows, cols, _ = subimg.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # Aplicar la transformación de afinidad
        rotated_image = cv2.warpAffine(subimg, rotation_matrix, (cols, rows))

        # Mostrar la imagen enderezada
        cv2.imshow('Imagen Enderezada', rotated_image)
        cv2.waitKey(0)

    else:
        print("No se detectaron líneas en la imagen.")

# Cerrar la ventana después de mostrar todas las imágenes
cv2.destroyAllWindows()


