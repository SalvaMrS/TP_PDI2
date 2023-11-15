import cv2
import numpy as np
import os
from skimage import measure
from skimage.filters import threshold_local
from skimage import segmentation
import matplotlib.pyplot as plt

# Cargar el modelo YOLO preentrenado y configuración
net = cv2.dnn.readNet("clasificador_patentes/model/yolov3.weights", "clasificador_patentes/model/yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Función para procesar una imagen y detectar patentes
def detect_and_segment_license_plate(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Preprocesamiento de la imagen
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Postprocesamiento para obtener las cajas delimitadoras
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # Clase 2 representa patentes en el modelo YOLO
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Eliminar cajas solapadas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indices = list(indices)

    # Mostrar resultados de detección
    for i in indices:
        try:
            x, y, w, h = boxes[i]

            # Ajustar coordenadas negativas
            x, y, w, h = map(max, (x, y, w, h), (0, 0, 0, 0))

            # Verificar que las dimensiones de la región de la patente sean válidas
            if 0 < w <= width and 0 < h <= height:
                plate_image = image[y:y + h, x:x + w]

                # Mostrar resultados
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Segmented License Plate', plate_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("La región de la patente no tiene dimensiones válidas.")
        except Exception as e:
            print(f"Error al procesar la imagen {image_path}: {str(e)}")
            print(f"Index en la lista de índices: {i}")
            continue


def segment_characters(license_plate_image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral local para obtener una imagen binaria
    thresh = threshold_local(gray, 21, offset=5, method="gaussian")
    binary = (gray > thresh).astype("uint8") * 255

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área para eliminar pequeños objetos no deseados
    min_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Crear una máscara en blanco
    mask = np.zeros_like(binary)

    # Dibujar los contornos en la máscara
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Segmentar los caracteres usando la máscara
    segmented_characters = cv2.bitwise_and(binary, mask)

    return segmented_characters, contours


# Carpeta que contiene las imágenes
carpeta_imagenes = 'Patentes/Patentes/'
archivos_en_carpeta = os.listdir(carpeta_imagenes)
extensiones_validas = ['.png']
archivos_imagen = [archivo for archivo in archivos_en_carpeta if any(archivo.endswith(ext) for ext in extensiones_validas)]

print(archivos_imagen)





# Procesar cada imagen
for imagen in archivos_imagen:

    imagen_path = os.path.join(carpeta_imagenes, imagen)
    license_plate_image = cv2.imread(imagen_path)

    # Segmentar caracteres
    segmented_characters, contours = segment_characters(license_plate_image)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
    plt.subplot(1, 3, 2), plt.imshow(segmented_characters, cmap='gray'), plt.title('Caracteres Segmentados')
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2RGB)), plt.title('Contornos')

    # Dibujar contornos en la imagen original
    contour_image = license_plate_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)), plt.title('Contornos')

    plt.show()
    #detect_and_segment_license_plate(imagen_path)
