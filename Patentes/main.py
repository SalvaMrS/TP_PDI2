import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = ".\\Patentes\\Imagenes\\"

# Lista para almacenar las imágenes cargadas
lista_imagenes = []

# Obtener la lista de archivos en la carpeta de imágenes
archivos_imagenes = os.listdir(carpeta_imagenes)

# Iterar a través de los archivos y cargar las imágenes en la lista
for archivo in archivos_imagenes:
    # Combinar la ruta de la carpeta con el nombre del archivo
    ruta_completa = os.path.join(carpeta_imagenes, archivo)

    # Leer la imagen usando OpenCV
    imagen = cv2.imread(ruta_completa)

    # Verificar si la lectura fue exitosa antes de agregarla a la lista
    if imagen is not None:
        lista_imagenes.append(imagen)
    else:
        print(f"No se pudo leer la imagen: {ruta_completa}")




def mostrar_imagenes_en_matriz(lista_imagenes, tipo_imagen='rgb'):
    num_filas = 3
    num_columnas = 4

    # Crear una figura con una matriz de subgráficos
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(12, 9))

    # Iterar a través de la lista de imágenes y mostrar cada una en un subgráfico
    for i in range(num_filas):
        for j in range(num_columnas):
            # Calcular el índice en la lista de imágenes
            indice_imagen = i * num_columnas + j

            # Verificar si hay imágenes restantes para mostrar
            if indice_imagen < len(lista_imagenes):
                # Mostrar la imagen en el subgráfico correspondiente según el tipo especificado
                if tipo_imagen.lower() == 'rgb':
                    axs[i, j].imshow(lista_imagenes[indice_imagen])
                elif tipo_imagen.lower() == 'hsv':
                    # Convertir a RGB antes de mostrar si el tipo es HSV
                    axs[i, j].imshow(cv2.cvtColor(lista_imagenes[indice_imagen], cv2.COLOR_HSV2RGB))
                elif tipo_imagen.lower() == 'gris':
                    axs[i, j].imshow(lista_imagenes[indice_imagen], cmap='gray')
                elif tipo_imagen.lower() == 'bin':
                    _, binarizada = cv2.threshold(cv2.cvtColor(lista_imagenes[indice_imagen], cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
                    axs[i, j].imshow(binarizada, cmap='gray')
                else:
                    print(f"Tipo de imagen no reconocido: {tipo_imagen}")
                    return
                
                axs[i, j].axis('off')  # Desactivar los ejes
            else:
                # Si no hay más imágenes, ocultar el subgráfico vacío
                axs[i, j].axis('off')

    # Ajustar el diseño y mostrar la figura
    plt.tight_layout()
    plt.show()


def transformar_imagenes(lista_imagenes, funcion_transformacion, variables={}):
    """
    Aplica una función de transformación a cada imagen en la lista.
    
    Args:
    - lista_imagenes: Lista de imágenes a transformar.
    - funcion_transformacion: Función que toma una imagen y variables como entrada y devuelve la imagen transformada.
    - variables: Diccionario de variables para la función de transformación (por defecto, es un diccionario vacío).
    
    Returns:
    - Lista de imágenes transformadas.
    """
    imagenes_transformadas = []
    
    for imagen in lista_imagenes:
        imagen_transformada = funcion_transformacion(imagen, **variables)
        imagenes_transformadas.append(imagen_transformada)
    
    return imagenes_transformadas


def filtro_sobel(imagen_entrada: np.ndarray, tamano_kernel: int = 3) -> np.ndarray:
    """
    Aplica el filtro Sobel en las direcciones x e y a la imagen dada.

    Parameters:
    - imagen_entrada (np.ndarray): La imagen a la cual se aplicará el filtro Sobel.
    - tamano_kernel (int, optional): Tamaño del kernel para el filtro Sobel. Por defecto es 3.

    Returns:
    - np.ndarray: La imagen resultante después de aplicar el filtro Sobel.
    """
    sobel_x = cv2.Sobel(imagen_entrada, cv2.CV_64F, 1, 0, ksize=tamano_kernel)
    sobel_y = cv2.Sobel(imagen_entrada, cv2.CV_64F, 0, 1, ksize=tamano_kernel)

    imagen_gradiente = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

    return imagen_gradiente


# Aplicar la conversión a escala de grises a cada imagen en la lista
lista_imagenes_grises = transformar_imagenes(lista_imagenes, cv2.cvtColor, {'code': cv2.COLOR_BGR2GRAY})

lista_imagenes_sobel = transformar_imagenes(lista_imagenes_grises, filtro_sobel)


def aplicar_umbral(imagen_gris, umbral):
    imagen_umbralizada = np.where(imagen_gris > umbral, imagen_gris, 0)
    return imagen_umbralizada.astype(np.uint8)

lista_imagenes_sobel_umbralizado = transformar_imagenes(lista_imagenes_sobel, aplicar_umbral, {'umbral': 100})

kernel_dilatacion = np.ones((4, 4), np.uint8)

lista_imagenes_sobel_umbralizado = transformar_imagenes(lista_imagenes_sobel_umbralizado, cv2.dilate, {'kernel':kernel_dilatacion})


# Definir el kernel para la erosión
kernel_erosion = np.ones((5, 1), np.uint8)

# Crear el diccionario de parámetros
parametros_erosion = {'kernel': kernel_erosion, 'iterations': 2}

# Aplicar erosión a cada imagen en la lista de imágenes Sobel umbralizadas
lista_imagenes_sobel_erocion = transformar_imagenes(lista_imagenes_sobel_umbralizado, cv2.erode, parametros_erosion)


# Definir el kernel para el filtro pasa bajo
x = 16
y = 4
kernel_pasa_bajo = np.ones((x, y), np.float32) / (x * y)

# Crear el diccionario de parámetros
parametros_filtro_pasa_bajo = {'ddepth': -1, 'kernel': kernel_pasa_bajo}

# Aplicar el filtro pasa bajo a cada imagen en la lista de imágenes Sobel erosionadas
lista_imagenes_sobel_filtro_pasa_bajo = transformar_imagenes(lista_imagenes_sobel_erocion, cv2.filter2D, parametros_filtro_pasa_bajo)

# Aplicar umbralización a cada imagen en la lista de imágenes Sobel filtradas con pasa bajo
umbral_binario = 220

lista_imagenes_sobel_binario = transformar_imagenes(lista_imagenes_sobel_filtro_pasa_bajo, aplicar_umbral, {'umbral': umbral_binario})


# Definir el kernel para la erosión
kernel_erosion2 = np.ones((1, 5), np.uint8)

# Crear el diccionario de parámetros
parametros_erosion2 = {'kernel': kernel_erosion2, 'iterations': 2}

# Aplicar erosión a cada imagen en la lista de imágenes Sobel umbralizadas
lista_imagenes_sobel_erocion2 = transformar_imagenes(lista_imagenes_sobel_binario, cv2.erode, parametros_erosion2)


# Definir el kernel elíptico para la clausura
kernel_eliptico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 22))

# Crear el diccionario de parámetros
parametros_clausura = {'op': cv2.MORPH_CLOSE,'kernel': kernel_eliptico, 'iterations': 1}

# Aplicar clausura a cada imagen en la lista de imágenes Sobel erosionadas
lista_imagenes_sobel_clausura = transformar_imagenes(lista_imagenes_sobel_erocion2, cv2.morphologyEx, parametros_clausura)



# Definir el umbral de tamaño mínimo para eliminar manchas
umbral_tamano_minimo = 200

# Función para eliminar manchas pequeñas en una imagen binaria
def eliminar_manchas_pequenas(imagen_binaria, umbral_tamano_minimo):
    # Encontrar contornos en la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara para mantener solo los contornos con un área mayor al umbral
    mascara_tamano_minimo = np.zeros_like(imagen_binaria)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area >= umbral_tamano_minimo:
            cv2.drawContours(mascara_tamano_minimo, [contorno], -1, 1, thickness=cv2.FILLED)

    # Aplicar la máscara para eliminar las manchas pequeñas
    imagen_filtrada = cv2.bitwise_and(imagen_binaria, imagen_binaria, mask=mascara_tamano_minimo)

    return imagen_filtrada

# Aplicar la función a cada imagen en la lista de imágenes Sobel clausuradas
lista_imagenes_sobel_filtrada = transformar_imagenes(lista_imagenes_sobel_clausura, eliminar_manchas_pequenas, {'umbral_tamano_minimo': umbral_tamano_minimo})



# Función para mantener solo la mancha más abajo de la imagen
def mantener_mancha_mas_abajo(imagen_binaria):
    # Encontrar contornos en la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno con el centroide más bajo
    centroide_mancha_mas_abajo = None
    y_maximo = 0

    for contorno in contornos:
        momentos = cv2.moments(contorno)
        if momentos["m00"] != 0:
            cx = int(momentos["m10"] / momentos["m00"])
            cy = int(momentos["m01"] / momentos["m00"])

            if cy > y_maximo:
                y_maximo = cy
                centroide_mancha_mas_abajo = contorno

    # Crear una imagen con la mancha más abajo
    imagen_mancha_mas_abajo = np.zeros_like(imagen_binaria)
    if centroide_mancha_mas_abajo is not None:
        cv2.drawContours(imagen_mancha_mas_abajo, [centroide_mancha_mas_abajo], -1, 1, thickness=cv2.FILLED)

    return imagen_mancha_mas_abajo

# Aplicar la función a cada imagen en la lista de imágenes Sobel filtradas
lista_imagenes_sobel_mancha_mas_abajo = transformar_imagenes(lista_imagenes_sobel_filtrada, mantener_mancha_mas_abajo)


kernel_dilatacion = np.ones((30, 20), np.uint8)

lista_imagenes_sobel_umbralizado2 = transformar_imagenes(lista_imagenes_sobel_mancha_mas_abajo, cv2.dilate, {'kernel':kernel_dilatacion})


# Crear una lista de tuplas con la imagen original y su correspondiente imagen umbralizada
tuplas_imagenes = list(zip(lista_imagenes, lista_imagenes_sobel_umbralizado2))

# Función para obtener la subimagen de la zona seleccionada de la imagen binarizada en la imagen original
def obtener_subimagen_zona_seleccionada(tupla):
    imagen_original, imagen_binaria = tupla

    # Encontrar contornos en la imagen binarizada
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno con el área más grande (zona blanca)
    contorno_seleccionado = max(contornos, key=cv2.contourArea, default=None)

    # Verificar si se encontró algún contorno
    if contorno_seleccionado is not None:
        # Crear una máscara para la zona seleccionada
        mascara_zona_seleccionada = np.zeros_like(imagen_binaria)
        cv2.drawContours(mascara_zona_seleccionada, [contorno_seleccionado], -1, 255, thickness=cv2.FILLED)

        # Aplicar la máscara para obtener la subimagen de la zona seleccionada en la imagen original
        subimagen_zona_seleccionada = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_zona_seleccionada)

        # Obtener las coordenadas del rectángulo delimitador de la zona seleccionada
        x, y, w, h = cv2.boundingRect(contorno_seleccionado)

        # Recortar la zona de interés de la subimagen
        zona_recortada = subimagen_zona_seleccionada[y:y+h, x:x+w]

        return zona_recortada
    else:
        # Si no se encontraron contornos, devolver una imagen en blanco
        return np.zeros_like(imagen_original)

# Aplicar la función a cada par de imágenes en la lista de tuplas
subimagenes_zona_seleccionada_recortada = transformar_imagenes(tuplas_imagenes, obtener_subimagen_zona_seleccionada)

# Mostrar el resultado con la función mostrar_imagenes_en_matriz
mostrar_imagenes_en_matriz(subimagenes_zona_seleccionada_recortada, 'gris')



