from typing import List, Callable, Dict, Tuple, Generator
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def debug():
    """
    Visualiza las imágenes de las variables globales que sean listas de imágenes.
    Utiliza la función mostrar_imagenes con detección automática del tipo de imagen.

    Returns:
    None
    """
    variables_globales = globals()
    for nombre_variable, valor_variable in variables_globales.items():
        if isinstance(valor_variable, list) and valor_variable and isinstance(valor_variable[0], np.ndarray):
            tipo_imagen = 'rgb' if valor_variable[0].shape[-1] == 3 else 'gris'
            print(f"Visualizando imágenes para la variable global: {nombre_variable}")
            mostrar_imagenes(valor_variable, tipo_imagen)




def cargar_imagenes_generador(carpeta_imagenes: str) -> Generator[np.ndarray, None, None]:
    """Generador que carga imágenes desde una carpeta dada.

    Parameters:
    - carpeta_imagenes (str): Ruta de la carpeta que contiene las imágenes.

    Yields:
    - np.ndarray: Imágenes cargadas como matrices NumPy.
    """
    archivos_imagenes = os.listdir(carpeta_imagenes)
    for archivo in archivos_imagenes:
        ruta_completa = os.path.join(carpeta_imagenes, archivo)

        imagen = cv2.imread(ruta_completa)
        imagen = cv2.medianBlur(imagen, 1)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        if imagen is not None:
            yield imagen
        else:
            print(f"No se pudo leer la imagen: {ruta_completa}")



def cargar_imagenes(carpeta_imagenes: str) -> List[np.ndarray]:
    """Carga imágenes desde una carpeta dada.

    Parameters:
    - carpeta_imagenes (str): Ruta de la carpeta que contiene las imágenes.

    Returns:
    - List[np.ndarray]: Lista de imágenes cargadas como matrices NumPy.
    """
    imagenes = []
    archivos_imagenes = os.listdir(carpeta_imagenes)
    for archivo in archivos_imagenes:
        ruta_completa = os.path.join(carpeta_imagenes, archivo)
        imagen = cv2.imread(ruta_completa)
        if imagen is not None:
            imagenes.append(imagen)
        else:
            print(f"No se pudo leer la imagen: {ruta_completa}")
    return imagenes

def mostrar_imagenes(imagenes: List[np.ndarray], tipo_imagen: str = 'rgb') -> None:
    """Muestra una matriz de imágenes en subgráficos.

    Parameters:
    - imagenes (List[np.ndarray]): Lista de imágenes a mostrar.
    - tipo_imagen (str): Tipo de representación de la imagen ('rgb', 'hsv', 'gris', 'bin').
                        Por defecto, es 'rgb'.

    Returns:
    - None
    """
    num_filas = 3
    num_columnas = 4
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(12, 9))
    for i in range(num_filas):
        for j in range(num_columnas):
            indice_imagen = i * num_columnas + j
            if indice_imagen < len(imagenes):
                if tipo_imagen.lower() == 'rgb':
                    axs[i, j].imshow(imagenes[indice_imagen])
                elif tipo_imagen.lower() == 'hsv':
                    axs[i, j].imshow(cv2.cvtColor(imagenes[indice_imagen], cv2.COLOR_HSV2RGB))
                elif tipo_imagen.lower() == 'gris':
                    axs[i, j].imshow(imagenes[indice_imagen], cmap='gray')
                elif tipo_imagen.lower() == 'bin':
                    _, binarizada = cv2.threshold(cv2.cvtColor(imagenes[indice_imagen], cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
                    axs[i, j].imshow(binarizada, cmap='gray')
                else:
                    print(f"Tipo de imagen no reconocido: {tipo_imagen}")
                    return
                
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def transformar_imagenes(imagenes: List[np.ndarray], funcion_transformacion: Callable[[np.ndarray, Dict], np.ndarray], variables: Dict = {}) -> List[np.ndarray]:
    """
    Aplica una función de transformación a cada imagen en la lista.
    
    Args:
    - imagenes (List[np.ndarray]): Lista de imágenes a transformar.
    - funcion_transformacion (Callable[[np.ndarray, dict], np.ndarray]): Función que toma una imagen y variables como entrada y devuelve la imagen transformada.
    - variables (dict): Diccionario de variables para la función de transformación (por defecto, es un diccionario vacío).
    
    Returns:
    - List[np.ndarray]: Lista de imágenes transformadas.
    """
    imagenes_transformadas = []
    
    for imagen in imagenes:
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

def aplicar_umbral(imagen_gris: np.ndarray, umbral: int) -> np.ndarray:
    """
    Aplica un umbral a una imagen en escala de grises.

    Parameters:
    - imagen_gris (np.ndarray): La imagen en escala de grises a la cual se aplicará el umbral.
    - umbral (int): Valor de umbral para binarizar la imagen.

    Returns:
    - np.ndarray: La imagen resultante después de aplicar el umbral.
    """
    imagen_umbralizada = np.where(imagen_gris > umbral, imagen_gris, 0)
    return imagen_umbralizada.astype(np.uint8)

def eliminar_objetos_pequenos(imagen_binaria: np.ndarray, umbral_tamano_minimo: int) -> np.ndarray:
    """
    Elimina objetos pequeñas en una imagen binaria basándose en un umbral de tamaño.

    Parameters:
    - imagen_binaria (np.ndarray): La imagen binaria de entrada.
    - umbral_tamano_minimo (int): El umbral de tamaño mínimo para mantener los contornos.

    Returns:
    - np.ndarray: La imagen binaria resultante después de eliminar las objetos pequeñas.
    """
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mascara_tamano_minimo = np.zeros_like(imagen_binaria)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area >= umbral_tamano_minimo:
            cv2.drawContours(mascara_tamano_minimo, [contorno], -1, 1, thickness=cv2.FILLED)

    imagen_filtrada = cv2.bitwise_and(imagen_binaria, imagen_binaria, mask=mascara_tamano_minimo)

    return imagen_filtrada

def mantener_objeto_mas_abajo(imagen_binaria):
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

def obtener_subimagen_zona_seleccionada(tupla: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Obtiene la subimagen de la zona seleccionada en la imagen original basándose en la imagen binaria.

    Parameters:
    - tupla (Tuple[np.ndarray, np.ndarray]): Una tupla que contiene la imagen original y la imagen binaria.

    Returns:
    - np.ndarray: La subimagen de la zona seleccionada en la imagen original.
    """
    imagen_original, imagen_binaria = tupla

    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contorno_seleccionado = max(contornos, key=cv2.contourArea, default=None)

    if contorno_seleccionado is not None:
        mascara_zona_seleccionada = np.zeros_like(imagen_binaria)
        cv2.drawContours(mascara_zona_seleccionada, [contorno_seleccionado], -1, 255, thickness=cv2.FILLED)
        subimagen_zona_seleccionada = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_zona_seleccionada)
        x, y, w, h = cv2.boundingRect(contorno_seleccionado)
        zona_recortada = subimagen_zona_seleccionada[y:y+h, x:x+w]

        return zona_recortada
    else:
        return np.zeros_like(imagen_original)