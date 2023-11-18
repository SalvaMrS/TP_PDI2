import cv2

from clasificador.Funciones import contador_monedas, monedas_y_dados, dibujar_moneda, \
    valor_moneda, imshow, agregar_total

# Cargar la imagen con contornos
ruta_imagen_contornos = './clasificador/imagenes/imagen_contornos.jpg'
imagen_contornos = cv2.imread(ruta_imagen_contornos, cv2.IMREAD_GRAYSCALE)

# Obtener contornos de monedas y dados
contornos_monedas, _ = monedas_y_dados(imagen_contornos)

# Cargar la imagen original
ruta_imagen_original = './clasificador/imagenes/monedas.jpg'
imagen_original = cv2.imread(ruta_imagen_original)

# Convertir la imagen a formato RGB
imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

# Iterar sobre los contornos de las monedas
for contorno_moneda in contornos_monedas:
    # Dibujar cada moneda en la imagen original
    imagen_original_rgb = dibujar_moneda(imagen_original_rgb, contorno_moneda, (0, 0, 0), valor_moneda(contorno_moneda))

# Calcular el total de las monedas
total_monedas = contador_monedas(contornos_monedas)

# Color del texto
color_texto = (255, 255, 255)

# Agregar texto a la izquierda de la imagen
imagen_resultante = agregar_total(imagen_original_rgb, total_monedas, color_texto)

# Guardar la imagen resultante
ruta_imagen_resultante = './clasificador/imagenes/monedas_resultado.jpg'
cv2.imwrite(ruta_imagen_resultante, cv2.cvtColor(imagen_resultante, cv2.COLOR_RGB2BGR))