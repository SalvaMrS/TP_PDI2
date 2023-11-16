import cv2
import numpy as np

imagen = cv2.imread('monedas.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > 100:  # Puedes ajustar este umbral según tu imagen
        cv2.drawContours(imagen, [contorno], -1, (0, 255, 0), 2)

cv2.imshow('Segmentación', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Problema 1b: Clasificación y conteo de monedas (requiere entrenamiento de modelo)
# Aquí necesitarías un modelo previamente entrenado para clasificar las monedas.

# Problema 1c: Determinación del número en cada dado (requiere entrenamiento de modelo)
# Similar al punto anterior, necesitarías un modelo entrenado para reconocer los números en los dados.
