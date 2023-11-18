import cv2
from clasificador.Funciones import monedas_y_dados, imagen_dado, valor_dado, dibujar_dado, imshow

ruta_imagen_contornos = './clasificador/imagenes/imagen_contornos.jpg'
imagen_contornos = cv2.imread(ruta_imagen_contornos, cv2.IMREAD_GRAYSCALE)

_, contornos_dados = monedas_y_dados(imagen_contornos)

ruta_imagen_resultado = './clasificador/imagenes/imagen_resultado.jpg'
imagen_resultado = cv2.imread(ruta_imagen_resultado)

imagen_resultado_rgb = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)

for contorno_dado in contornos_dados:
    imagen_original_gris = cv2.imread('./clasificador/imagenes/monedas.jpg', cv2.IMREAD_GRAYSCALE)

    imagen_recortada = imagen_dado(imagen_original_gris, contorno_dado)

    puntos_dado = valor_dado(imagen_recortada)

    imagen_resultado_rgb = dibujar_dado(imagen_resultado_rgb, contorno_dado, (0, 0, 0), puntos_dado)


cv2.imwrite(ruta_imagen_resultado, cv2.cvtColor(imagen_resultado_rgb, cv2.COLOR_RGB2BGR))