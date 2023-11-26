import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from Funciones import *

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
kernel_dilatacion = np.ones((15, 35), np.uint8)
imagenes_sobel_umbralizado2 = transformar_imagenes(imagenes_sobel_mancha_mas_abajo, cv2.dilate, {'kernel':kernel_dilatacion})
tuplas_imagenes = list(zip(imagenes, imagenes_sobel_umbralizado2))
subimagenes_zona_seleccionada = transformar_imagenes(tuplas_imagenes, obtener_subimagen_zona_seleccionada)
for subimg in subimagenes_zona_seleccionada:
    subimg_gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    _, subimg_humbr = cv2.threshold(subimg_gray, 90, 255, cv2.THRESH_OTSU)
    threshol = _ +(_*0.12)
    _, subimg_humbr = cv2.threshold(subimg_gray, _ + 20 , 255, cv2.THRESH_BINARY)


    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(subimg_humbr)

    # Imprime la cantidad de componentes encontrados
    print("Número total de componentes conectados:", retval)

    cv2.imshow('Imagen Original', subimg)

    # Imprime las estadísticas de cada componente
    for i in range(1, retval):  # Comienza desde 1 para omitir el componente de fondo (etiqueta 0)
        obj = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar por área mínima
        area = cv2.contourArea(contours[0])
        if area < 5 or area > 340:
            continue

        # Aproximación de forma
        epsilon = 0.04 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Filtrar por forma (por ejemplo, número de lados)
        if len(approx) > 3:  # Asumiendo que estás buscando letras y números

            obj_color = cv2.merge((obj, obj, obj))
            cv2.drawContours(obj_color, [approx], -1, (255, 0, 0), 1)

            st = stats[i, :]
            cv2.rectangle(obj_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 0, 255), thickness=1)

            cv2.imshow('Objeto Filtrado', obj_color)
            cv2.waitKey(0)

    # Espera la entrada del teclado antes de pasar a la siguiente imagen
    cv2.waitKey(0)
cv2.destroyAllWindows()


