import  os
import cv2
from clasificador_patentes.ocr import PatenteOCR
ocr_detector = PatenteOCR()

# Ruta de la imagen que deseas procesar

carpeta_imagenes = 'Patentes/Patentes/'
archivos_en_carpeta = os.listdir(carpeta_imagenes)
extensiones_validas = ['.png','.jpeg', '.jpeg','.webp']
archivos_imagen = [archivo for archivo in archivos_en_carpeta if any(archivo.endswith(ext) for ext in extensiones_validas)]
for imagen in archivos_imagen:
    imagen_path = os.path.join(carpeta_imagenes, imagen)
    frame = cv2.imread(imagen_path)
    # Obtener las dimensiones de la imagen
    height, width, _ = frame.shape
    # Definir coordenadas que cubran toda la imagen
    # En este caso, se usa (0, 0, width, height) para cubrir toda la imagen
    iter_coords = [(0, 0, width, height, 0.8)]  # Coordenadas que cubren toda la imagen
    # Realizar la predicción de la patente
    patentes_detectadas = ocr_detector.predict(iter_coords, frame)
    print(patentes_detectadas, imagen_path)
    # Imprimir los valores de las patentes detectadas
    for i, patente in enumerate(patentes_detectadas, 1):
        print(f"Patente {i}: {patente}")