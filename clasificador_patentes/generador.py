import cv2
import os
import logging

class GeneradorDeImagenes:
    def __init__(self, output_folder='/Procesadas'):
        self.output_folder = 'Patentes/Procesadas/'
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def guardar_subimagenes(self, subimagenes):
        for i, subimg in enumerate(subimagenes):
            filename = self._generar_nombre_archivo(i)
            self._guardar_imagen(filename, subimg)
            self.logger.info(f'Se ha guardado {filename}')

    def _generar_nombre_archivo(self, index):
        return os.path.join(self.output_folder, f'{index}.jpg')

    def _guardar_imagen(self, filename, imagen):
        cv2.imwrite(filename, imagen)
