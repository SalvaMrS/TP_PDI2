import cv2

from clasificador.Funciones import contador_monedas, monedas_y_dados, dibujar_moneda, \
    valor_moneda, imshow, agregar_total

img = cv2.imread('./clasificador/imagenes/imagen_contornos.jpg', cv2.IMREAD_GRAYSCALE)

monedas, _ = monedas_y_dados(img)

img_real = cv2.imread('./clasificador/imagenes/monedas.jpg')

img_rgb = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)

for i in monedas:
    img_rgb = dibujar_moneda(img_rgb, i, (0, 0, 0), valor_moneda(i))

total_monedas = contador_monedas(monedas)

color_texto = (255, 255, 255)

# Agregar texto a la izquierda de la imagen
img_resultante = agregar_total(img_rgb, f'TOTAL: ${total_monedas}', total_monedas, color_texto)

# Guardar la imagen resultante
cv2.imwrite('./clasificador/imagenes/monedas_resultado.jpg', cv2.cvtColor(img_resultante, cv2.COLOR_RGB2BGR))