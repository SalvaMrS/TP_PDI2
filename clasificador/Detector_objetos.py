import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=True):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

# Ruta de la imagen
ruta_imagen = 'monedas.jpg'

# Lee la imagen en formato RGB
img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Aplicar suavizado para reducir el ruido antes de aplicar Canny
img_suavizada = cv2.GaussianBlur(img, (15, 15), 0)

# Aplicar el filtro Sobel en las direcciones x e y
sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud del gradiente
img_g = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))


img_fil_umbral = cv2.GaussianBlur(img_g, (19, 19), 0)

# Convertir la magnitud del gradiente a tipo CV_8U
img_g = cv2.convertScaleAbs(img_fil_umbral)


# Calcula el umbral como el 90% del valor máximo de píxel
umbral = int(np.percentile(img_g, 90))

# Binariza la imagen
_, img_bin = cv2.threshold(img_g, umbral, 1, cv2.THRESH_BINARY)

# Define el kernel para la operación morfológica
kernel_erosion = np.ones((5, 5), np.uint8)

kernel_expansion = np.ones((5, 5), np.uint8)

expansion = img_bin.copy()

# Aplica una apertura para obtener la "erosión"
expansion = cv2.morphologyEx(expansion, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
erosion = cv2.morphologyEx(expansion, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))

for _ in range(50):
    # Aplica una apertura para obtener la "erosión"
    erosion = cv2.morphologyEx(expansion, cv2.MORPH_OPEN, kernel_erosion)

    # Aplica un cierre para obtener la "expansión"
    expansion = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_expansion)


# Encuentra los contornos en la imagen binarizada
contornos, _ = cv2.findContours(expansion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibuja los contornos en una copia de la imagen binarizada original
img_contornos = img_bin.copy()

# Descarta contornos con un área menor a 1000 píxeles
for i, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)
    if area >= 1000:
        cv2.drawContours(img_contornos, [contorno], -1, 255, thickness=cv2.FILLED)


radio_circulo = 35
kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radio_circulo + 1, 2*radio_circulo + 1))

img_contornos = cv2.morphologyEx(img_contornos, cv2.MORPH_OPEN, kernel_cierre)


# Muestra la imagen binarizada original con los contornos descartados
plt.subplot(1, 2, 1), plt.imshow(expansion, cmap='gray')
plt.title('Imagen Binarizada'), plt.xticks([]), plt.yticks([])

# Muestra la imagen con los contornos descartados
plt.subplot(1, 2, 2), plt.imshow(img_contornos, cmap='gray')
plt.title('Contornos Filtrados'), plt.xticks([]), plt.yticks([])

plt.show()
