from clasificador import Detector_objetos, Contador_Monedas, Lector_dados
from clasificador.Funciones import imshow
import cv2


def main():
    Detector_objetos  # Esto importa y ejecuta el código en Detector_objetos.py
    Contador_Monedas
    Lector_dados

    ruta = './clasificador/imagenes/imagen_resultado.jpg'
    imagen= cv2.imread(ruta)

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imshow(imagen)



# Verifica si este script es el punto de entrada principal
if __name__ == "__main__":
    main()