import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from Funciones import *

carpeta_imagenes = ".\\Patentes\\Imagenes\\"
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

debug()