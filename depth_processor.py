"""Модуль для обработки карт глубины"""
import numpy as np
import cv2
from config import Config

class DepthProcessor:
    @staticmethod
    def remove_person_from_depth(depth_image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Удаление людей из карты глубины"""
        cleaned_depth = depth_image.copy()
        
        # Минимальное расширение маски
        kernel = np.ones((3, 3), np.uint8)
        person_mask_dilated = cv2.dilate(person_mask, kernel, iterations=1)
        
        # Обнуление областей с людьми
        cleaned_depth[person_mask_dilated > 0] = 0
        
        return cleaned_depth
    
    @staticmethod
    def create_colormap(depth_image: np.ndarray) -> np.ndarray:
        """Создание цветовой карты глубины"""
        return cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=Config.DEPTH_SCALE), 
            cv2.COLORMAP_JET
        )
