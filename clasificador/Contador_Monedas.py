import cv2
import numpy as np
from clasificador.Funciones import contador_monedas, monedas_y_dados, dibujar_moneda, \
    valor_moneda, imshow, agregar_total, contar_monedas_por_valor

ruta_imagen_contornos = './clasificador/imagenes/imagen_contornos.jpg'
imagen_contornos = cv2.imread(ruta_imagen_contornos, cv2.IMREAD_GRAYSCALE)

contornos_monedas, _ = monedas_y_dados(imagen_contornos)

ruta_imagen_original = './clasificador/imagenes/monedas.jpg'
imagen_original = cv2.imread(ruta_imagen_original)

imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

for contorno_moneda in contornos_monedas:
    imagen_original_rgb = dibujar_moneda(imagen_original_rgb, contorno_moneda, (0, 0, 0), valor_moneda(contorno_moneda))

total_monedas = contador_monedas(contornos_monedas)

"""imagen_resultante = agregar_total(imagen_original_rgb, total_monedas, (255, 255, 255))

ruta_imagen_resultante = './clasificador/imagenes/imagen_resultado.jpg'
cv2.imwrite(ruta_imagen_resultante, cv2.cvtColor(imagen_resultante, cv2.COLOR_RGB2BGR))
"""
cantidades_por_valor = contar_monedas_por_valor(contornos_monedas)
imagen_resultante = agregar_total(imagen_original_rgb, total_monedas, (255, 255, 255))

ruta_imagen_resultante = './clasificador/imagenes/imagen_resultado.jpg'
cv2.imwrite(ruta_imagen_resultante, cv2.cvtColor(imagen_resultante, cv2.COLOR_RGB2BGR))

print("Cantidad de monedas de 10 centavos:", cantidades_por_valor.get('10'))
print("Cantidad de monedas de 50 centavos:", cantidades_por_valor.get('50'))
print("Cantidad de monedas de 1 peso:", cantidades_por_valor.get('1'))