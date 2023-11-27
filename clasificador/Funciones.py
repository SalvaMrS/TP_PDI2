import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img: np.ndarray, title: str = None, color_img: bool = False, blocking: bool = True) -> None:
    """
    Muestra una imagen utilizando Matplotlib.

    Parameters:
    - img (np.ndarray): La imagen que se mostrará.
    - title (str, optional): El título de la imagen. Por defecto es None.
    - color_img (bool, optional): Indica si la imagen es a color. Por defecto es False.
    - blocking (bool, optional): Indica si la ejecución del programa se bloquea hasta que se cierra la ventana de la imagen. Por defecto es True.
    """
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

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

def apertura(img: np.ndarray, radio_circulo: int = 35) -> np.ndarray:
    """
    Aplica la operación de apertura a una imagen utilizando un kernel circular.

    Parameters:
    - img (np.ndarray): La imagen de entrada.
    - radio_circulo (int): El radio del elemento estructurante circular. Por defecto, se establece en 35.

    Returns:
    - np.ndarray: La imagen resultante después de aplicar la operación de apertura.
    """
    kernel_apertura  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio_circulo + 1, 2 * radio_circulo + 1))

    imagen_contornos = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_apertura)

    return imagen_contornos

def obtener_contornos(imagen_entrada: np.ndarray) -> np.ndarray:
    """
    Aplica una serie de operaciones para obtener y resaltar los contornos en una imagen.

    Parameters:
    - imagen_entrada (np.ndarray): La imagen de entrada.

    Returns:
    - np.ndarray: La imagen resultante después de resaltar los contornos.
    """

    imagen_suavizada = cv2.GaussianBlur(imagen_entrada, (15, 15), 0)

    imagen_gradiente = filtro_sobel(imagen_suavizada)

    imagen_filtrada_umbral = cv2.GaussianBlur(imagen_gradiente, (19, 19), 0)

    imagen_gradiente_abs = cv2.convertScaleAbs(imagen_filtrada_umbral)

    umbral_superior = int(np.percentile(imagen_gradiente_abs, 90))

    _, imagen_umbralizada = cv2.threshold(imagen_gradiente_abs, umbral_superior, 1, cv2.THRESH_BINARY)

    imagen_expandida = cv2.morphologyEx(imagen_umbralizada, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contornos, _ = cv2.findContours(imagen_expandida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contorno in enumerate(contornos):
        area_contorno = cv2.contourArea(contorno)
        if area_contorno >= 1000:
            cv2.drawContours(imagen_umbralizada, [contorno], -1, 255, thickness=cv2.FILLED)

    imagen_contornos = apertura(imagen_umbralizada)

    return imagen_contornos

def es_circulo(contorno: np.ndarray) -> bool:
    """
    Determina si un contorno se asemeja a la forma de un círculo basándose en su factor de forma.

    Parameters:
    - contorno (np.ndarray): El contorno para evaluar.

    Returns:
    - bool: True si el contorno se asemeja a un círculo (factor de forma > 0.8), False en caso contrario.
    """
    area_contorno = cv2.contourArea(contorno)
    perimetro_contorno = cv2.arcLength(contorno, True)
    factor_forma = 4 * np.pi * area_contorno / (perimetro_contorno ** 2)

    return factor_forma > 0.8

def valor_moneda(contorno: np.ndarray) -> float:
    """
    Asigna un valor a un contorno basándose en el área del contorno, representando el valor de una moneda.

    Parameters:
    - contorno (np.ndarray): El contorno para evaluar.

    Returns:
    - float: El valor asignado a la moneda según su área.
    """
    area_contorno = cv2.contourArea(contorno)

    if area_contorno < 80000:
        return 0.1
    elif area_contorno < 105000:
        return 1
    else:
        return 0.5

def monedas_y_dados(imagen_binaria: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Identifica y clasifica los contornos de monedas y dados en una imagen binaria.

    Parameters:
    - img (np.ndarray): La imagen binaria de entrada.

    Returns:
    - tuple[list[np.ndarray], list[np.ndarray]]: Una tupla con dos listas, la primera contiene los contornos de monedas y la segunda los contornos de dados.
    """
    _, imagen_umbralizada = cv2.threshold(imagen_binaria, 127, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_monedas = []
    contornos_dados = []

    for contorno in contornos:
        if es_circulo(contorno):
            contornos_monedas.append(contorno)
        else:
            contornos_dados.append(contorno)

    return contornos_monedas, contornos_dados

def contador_monedas(contornos_monedas: list[np.ndarray]) -> float:
    """
    Calcula el valor total de las monedas en la lista proporcionada.

    Parameters:
    - contornos_monedas (list[np.ndarray]): Lista de contornos de monedas.

    Returns:
    - float: El valor total de las monedas redondeado a una décima.
    """
    valor_total = 0

    for contorno_moneda in contornos_monedas:
        if es_circulo(contorno_moneda):
            valor_total += valor_moneda(contorno_moneda)

    return round(valor_total, 1)

def contar_monedas_por_valor(contornos_monedas):
    cantidades = {'10': 0, '50': 0, '1': 0}

    for contorno_moneda in contornos_monedas:
        valor = valor_moneda(contorno_moneda)
        if valor == 0.1:
            cantidades['10'] += 1
        elif valor == 0.5:
            cantidades['50'] += 1
        elif valor == 1:
            cantidades['1'] += 1

    return cantidades

def dibujar_moneda(imagen_original: np.ndarray, contorno_moneda: np.ndarray, color: tuple[int, int, int], valor_moneda: int) -> np.ndarray:
    """
    Dibuja el contorno de una moneda con su valor dentro en una copia de la imagen original.

    Parameters:
    - imagen_original (np.ndarray): Imagen original sobre la cual dibujar el contorno.
    - contorno_moneda (np.ndarray): Contorno de la moneda.
    - color (tuple[int, int, int]): Color del contorno y del texto.
    - valor (int): Valor numérico de la moneda.

    Returns:
    np.ndarray: Una copia de la imagen original con el contorno de la moneda dibujado y el valor etiquetado.
    """

    imagen_resultante = imagen_original.copy()

    x, y, ancho, alto = cv2.boundingRect(contorno_moneda)

    cv2.drawContours(imagen_resultante, [contorno_moneda], 0, color, 5)

    tamano_texto = max(ancho, alto) // 2

    centro_texto = ((2 * x + ancho) // 2, (2 * y + alto) // 2)

    posicion_texto = (centro_texto[0] - tamano_texto // 4, centro_texto[1] + tamano_texto // 4)

    cv2.putText(imagen_resultante, str(valor_moneda), posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, tamano_texto / 50, color, 10, cv2.LINE_AA)

    return imagen_resultante

def dibujar_dado(imagen_original: np.ndarray, contorno_dado: list[np.ndarray], color: tuple[int, int, int], valor_dado: int) -> np.ndarray:
    """
    Dibuja el contorno de un dado en una imagen y agrega el valor del dado en la esquina inferior izquierda.

    Parameters:
    - img_original (numpy.ndarray): Imagen original sobre la que se va a dibujar.
    - contorno_dado (list[numpy.ndarray]): Contorno del dado.
    - color (tuple[int, int, int]): Color del contorno y del valor del dado.
    - valor (int): Valor numérico del dado.

    Returns:
    - img_resultado (numpy.ndarray): Imagen resultante con el contorno y el valor del dado dibujados.
    """
    imagen_resultante = imagen_original.copy()

    x, y, ancho, alto = cv2.boundingRect(contorno_dado)

    cv2.drawContours(imagen_resultante, [contorno_dado], 0, color, 5)

    tamano_texto = max(ancho, alto)

    posicion_texto = (x + ancho - 50, y + alto + 50)

    cv2.putText(imagen_resultante, str(valor_dado), posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, tamano_texto / 50, color, 10, cv2.LINE_AA)

    return imagen_resultante

def agregar_total(imagen_original: np.ndarray, valor_numerico: str, color: tuple[int, int, int]) -> np.ndarray:
    """
    Agrega un texto con un valor total en la esquina inferior izquierda de la imagen.

    Parameters:
    - imagen_original (numpy.ndarray): Imagen original sobre la que se va a agregar el texto.
    - texto_descriptivo (str): Texto descriptivo.
    - valor_numerico (str): Valor numérico a mostrar.
    - color (tuple[int, int, int]): Color del texto.

    Returns:
    - imagen_resultante (numpy.ndarray): Imagen resultante con el texto y valor agregados.
    """
    imagen_resultante = imagen_original.copy()

    dimensiones_texto = cv2.getTextSize('Total', cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]

    posicion_texto = (10, imagen_original.shape[0] - 10)

    cv2.putText(imagen_resultante, f"TOTAL: ${valor_numerico}", posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10, cv2.LINE_AA)

    return imagen_resultante

def imagen_dado(imagen_original: np.ndarray, contorno_dado: list[np.ndarray]) -> np.ndarray:
    """
    Recorta la región de interés dentro de un contorno en una imagen y la devuelve.

    Parameters:
    - imagen_original (numpy.ndarray): Imagen original de la cual se recortará la región de interés.
    - contorno_dado (list[numpy.ndarray]): Contorno del dado.

    Returns:
    - region_interes_recortada (numpy.ndarray): Región de interés recortada dentro del contorno.
    """
    mascara = np.zeros_like(imagen_original, dtype=np.uint8)

    cv2.drawContours(mascara, [contorno_dado], 0, (255, 255, 255), thickness=cv2.FILLED)

    x, y, ancho, alto = cv2.boundingRect(contorno_dado)

    mascara_inversa = cv2.bitwise_not(mascara)

    imagen_resultante = cv2.bitwise_or(imagen_original, mascara_inversa)

    region_interes_recortada = imagen_resultante[y:y+alto, x:x+ancho]

    return region_interes_recortada

def valor_dado(imagen: np.ndarray) -> int:
    """
    Cuenta la cantidad de puntos en una imagen con un dado.

    Parameters:
    - imagen (numpy.ndarray): Imagen en la que se contarán los puntos.

    Returns:
    - valor (int): Cantidad de puntos encontrados en la imagen.
    """
    imagen_suavizada = cv2.GaussianBlur(imagen, (55, 55), 0)

    porcentaje_brillantes = 10
    umbral_brillantes = np.percentile(imagen_suavizada, porcentaje_brillantes)

    _, imagen_umbral = cv2.threshold(imagen_suavizada, umbral_brillantes, 255, cv2.THRESH_BINARY)

    imagen_binaria_invertida = cv2.bitwise_not(imagen_umbral)

    contornos, _ = cv2.findContours(imagen_binaria_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valor = 0

    for contorno in contornos:
        area_contorno = cv2.contourArea(contorno)
        umbral_area = 1700
        if es_circulo(contorno) and area_contorno > umbral_area:
            valor += 1

    return valor
