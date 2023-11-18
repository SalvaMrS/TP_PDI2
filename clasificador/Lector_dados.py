import cv2
from clasificador.Funciones import monedas_y_dados, imagen_dado, valor_dado, dibujar_dado, imshow

# Cargar la imagen con contornos
ruta_imagen_contornos = './clasificador/imagenes/imagen_contornos.jpg'
imagen_contornos = cv2.imread(ruta_imagen_contornos, cv2.IMREAD_GRAYSCALE)

# Obtener los contornos de monedas y dados
_, contornos_dados = monedas_y_dados(imagen_contornos)

# Cargar la imagen resultante
ruta_imagen_resultado = './clasificador/imagenes/monedas_resultado.jpg'
imagen_resultado = cv2.imread(ruta_imagen_resultado)

# Convertir la imagen a formato RGB
imagen_resultado_rgb = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)

# Iterar sobre los contornos de los dados
for contorno_dado in contornos_dados:
    # Cargar la imagen original en escala de grises
    imagen_original_gris = cv2.imread('./clasificador/imagenes/monedas.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Recortar la región de interés
    imagen_recortada = imagen_dado(imagen_original_gris, contorno_dado)
    
    # Calcular los puntos en el dado
    puntos_dado = valor_dado(imagen_recortada)

    # Supongamos que 'contorno_dado' está definido y 'color' es un tuple (B, G, R)
    imagen_resultado_rgb = dibujar_dado(imagen_resultado_rgb, contorno_dado, (0, 0, 0), puntos_dado)

# Guardar la imagen resultante
cv2.imwrite(ruta_imagen_resultado, cv2.cvtColor(imagen_resultado_rgb, cv2.COLOR_RGB2BGR))