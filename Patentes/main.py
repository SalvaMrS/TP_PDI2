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

kernel_dilatacion = np.ones((2, 2), np.uint8)

lista_imagenes_sobel_umbralizado = transformar_imagenes(lista_imagenes_sobel_umbralizado, cv2.dilate, {'kernel':kernel_dilatacion})

# Mostrar las imágenes umbralizadas
mostrar_imagenes_en_matriz(lista_imagenes_sobel_umbralizado, 'gris')



# Aplicar el operador Canny a cada imagen en la lista
lista_bordes_canny = transformar_imagenes(lista_imagenes_grises, cv2.Canny, {'threshold1':20, 'threshold2':150, 'apertureSize': 3})



# mostrar_imagenes_en_matriz(lista_bordes_canny, 'gris')




# Aplicar dilatación a cada imagen en la lista de bordes Canny
kernel_dilatacion = np.ones((1, 1), np.uint8)
lista_bordes_canny_dilatados = transformar_imagenes(lista_bordes_canny, cv2.dilate, {'kernel':kernel_dilatacion})

# mostrar_imagenes_en_matriz(lista_bordes_canny_dilatados, 'gris')






# Aplicar la Transformada de Hough para encontrar líneas en una imagen Canny
def encontrar_lineas(imagen_canny):
    lineas = cv2.HoughLines(imagen_canny, rho=1, theta=np.pi / 180, threshold=100)

    # Crear una imagen en blanco con el mismo tamaño que la imagen Canny
    imagen_lineas = np.zeros_like(imagen_canny)

    for linea in lineas:
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Dibujar la línea en la imagen de líneas
        cv2.line(imagen_lineas, (x1, y1), (x2, y2), 1, 2)

    return imagen_lineas



# imagenes_lineas = transformar_imagenes(lista_bordes_canny_dilatados, encontrar_lineas)

# mostrar_imagenes_en_matriz(imagenes_lineas, 'gris')






# Encontrar todos los contornos en cada imagen de la lista de bordes Canny dilatados
lista_contornos = transformar_imagenes(lista_bordes_canny_dilatados, cv2.findContours, {'mode': cv2.RETR_LIST, 'method': cv2.CHAIN_APPROX_SIMPLE})

# Obtener solo los contornos de cada imagen de la lista
lista_contornos = [contorno[0] for contorno in lista_contornos]

def dibujar_contornos(imagen, lista_contornos):
    """
    Dibuja los contornos en una imagen.

    Args:
    - imagen: Imagen en la que se dibujarán los contornos.
    - lista_contornos: Lista de contornos a dibujar en la imagen.

    Returns:
    - Imagen resultante con los contornos dibujados.
    """
    # Copiar la imagen original para no modificar la original
    imagen_con_contornos = imagen.copy()

    # Dibujar los contornos en la imagen copiada
    cv2.drawContours(imagen_con_contornos, lista_contornos, -1, (0, 255, 0), 2)

    return imagen_con_contornos


imagenes_contornos = transformar_imagenes(lista_imagenes, dibujar_contornos, {'lista_contornos': lista_contornos[2]})

# mostrar_imagenes_en_matriz(lista_bordes_canny_dilatados, 'gris')
# mostrar_imagenes_en_matriz(imagenes_contornos)

