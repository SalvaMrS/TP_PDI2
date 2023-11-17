import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title=None, color_img=False, blocking=True):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

def sobel(img, ksize=3):
    # Aplicar el filtro Sobel en las direcciones x e y
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calcular la magnitud del gradiente
    img_g = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

    return img_g

def apertura(img, radio_circulo = 35):
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radio_circulo + 1, 2*radio_circulo + 1))

    img_contornos = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_cierre)

    return img_contornos

def obtener_contornos(img):
    # Aplicar suavizado para reducir el ruido antes de aplicar Canny
    img_suavizada = cv2.GaussianBlur(img, (15, 15), 0)

    img_g = sobel(img_suavizada)

    img_fil_umbral = cv2.GaussianBlur(img_g, (19, 19), 0)

    # Convertir la magnitud del gradiente a tipo CV_8U
    img_g = cv2.convertScaleAbs(img_fil_umbral)

    # Calcula el umbral como el 90% del valor máximo de píxel
    umbral = int(np.percentile(img_g, 90))

    # Binariza la imagen
    _, img_bin = cv2.threshold(img_g, umbral, 1, cv2.THRESH_BINARY)

    # Expansion
    expansion = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Encuentra los contornos en la imagen binarizada
    contornos, _ = cv2.findContours(expansion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Descarta contornos con un área menor a 1000 píxeles
    for i, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        if area >= 1000:
            cv2.drawContours(img_bin, [contorno], -1, 255, thickness=cv2.FILLED)

    img_contornos = apertura(img_bin)

    return img_contornos

def es_circulo(contour):
    # Calcular área, perímetro y factor de forma
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    form_factor = 4 * np.pi * area / (perimeter ** 2)

    return form_factor > 0.8

def valor_moneda(contour):
    area = cv2.contourArea(contour)

    if area < 80000:
        return 0.1
    elif area < 105000:
        return 1
    else:
        return 0.5

def monedas_y_dados(img):

    _, img_binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar listas para contornos de monedas y dados
    contornos_monedas = []
    contornos_dados = []

    # Iterar sobre los contornos
    for contour in contours:
        # Determinar si es un círculo
        if es_circulo(contour):
            contornos_monedas.append(contour)
        else:
            contornos_dados.append(contour)

    return contornos_monedas, contornos_dados

def contador_monedas(monedas):
    valor = 0
    for coin in monedas:
        if es_circulo(coin):
            valor += valor_moneda(coin)
    
    return round(valor,1)

def dibujar_moneda(img_original, contorno, color, valor):
    # Crear una copia de la imagen original
    img_resultado = img_original.copy()

    # Obtener el rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contorno)

    # Dibujar el contorno exterior en la imagen resultado con grosor adicional
    cv2.drawContours(img_resultado, [contorno], 0, color, 5)

    # Calcular el tamaño del texto en función del tamaño del contorno
    tamano_texto = max(w, h) // 2

    # Obtener la posición para colocar el valor centrado dentro del contorno
    centro_texto = ((2 * x + w) // 2, (2 * y + h) // 2)

    # Calcular la posición del texto centrado
    posicion_texto = (centro_texto[0] - tamano_texto // 4, centro_texto[1] + tamano_texto // 4)

    # Dibujar el valor dentro del contorno con el tamaño ajustado
    cv2.putText(img_resultado, str(valor), posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, tamano_texto / 50, color, 10, cv2.LINE_AA)

    return img_resultado

def dibujar_dado(img_original, contorno, color, valor):
    # Crear una copia de la imagen original
    img_resultado = img_original.copy()

    # Obtener el rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contorno)

    # Dibujar el contorno exterior en la imagen resultado
    cv2.drawContours(img_resultado, [contorno], 0, color, 5)

    # Calcular el tamaño del texto en función del tamaño del contorno
    tamano_texto = max(w, h)

    # Obtener la posición para colocar el valor en la esquina inferior izquierda
    posicion_texto = (x + w - 50, y + h + 50)

    # Dibujar el valor en la esquina inferior izquierda
    cv2.putText(img_resultado, str(valor), posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, tamano_texto / 50, color, 10, cv2.LINE_AA)

    return img_resultado


def agregar_total(imagen, texto, valor, color):
    # Copiar la imagen original para no modificar la original
    img_resultado = imagen.copy()

    # Configurar la fuente y el tamaño del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    tamano_texto = 5
    grosor_texto = 10

    # Obtener las dimensiones del texto
    dimensiones_texto = cv2.getTextSize(texto, font, tamano_texto, grosor_texto)[0]

    # Definir la posición del texto (abajo a la izquierda)
    posicion_texto = (10, imagen.shape[0] - 10)

    # Dibujar el texto en la imagen
    cv2.putText(img_resultado, texto, posicion_texto, font, tamano_texto, color, grosor_texto, cv2.LINE_AA)

    return img_resultado

def imagen_dado(imagen_original, contorno):

    # Crear una máscara negra del mismo tamaño que la imagen original
    mask = np.zeros_like(imagen_original, dtype=np.uint8)

    # Dibujar el contorno en blanco en la máscara
    cv2.drawContours(mask, [contorno], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Encontrar las coordenadas del rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contorno)

    # Crear la máscara inversa (fuera del área del contorno)
    mask_inversa = cv2.bitwise_not(mask)

    # Hacer que la zona fuera del área del contorno sea blanca en la imagen original
    imagen_resultante = cv2.bitwise_or(imagen_original, mask_inversa)

    # Recortar la región de interés dentro del contorno
    region_interes_recortada = imagen_resultante[y:y+h, x:x+w]

    return region_interes_recortada

def cant_puntos(imagen):
    imagen = cv2.GaussianBlur(imagen, (55, 55), 0)

    # Obtener el valor del umbral como el 10% de los píxeles más brillantes
    porcentaje = 10
    umbral_brillantes = np.percentile(imagen, porcentaje)

    # Binarizar la imagen con el nuevo umbral
    _, imagen_bin = cv2.threshold(imagen, umbral_brillantes, 255, cv2.THRESH_BINARY)

    # Invertir la imagen binarizada
    imagen_bin_invertida = cv2.bitwise_not(imagen_bin)

    # Encontrar contornos en la imagen binarizada invertida
    contornos, _ = cv2.findContours(imagen_bin_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valor = 0

    for i in contornos:
        area = cv2.contourArea(i)
        umbral = 1700
        if es_circulo(i) and area > umbral:
            valor += 1

    return valor