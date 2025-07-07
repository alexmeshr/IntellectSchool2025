"""Модуль для обработки карт глубины"""
import numpy as np
import cv2
from config import Config

class DepthProcessor:
    @staticmethod
    def keep_only_baggage(depth_image: np.ndarray, baggage_mask: np.ndarray) -> np.ndarray:
        """Сохраняет только багаж в карте глубины, обнуляя все остальное"""
        # Создаем копию глубинного изображения
        cleaned_depth = depth_image.copy()
        
        # Обнуляем все области, где маска багажа равна 0 (не багаж)
        cleaned_depth[baggage_mask == 0] = 0
        
        return cleaned_depth
    
    @staticmethod
    def create_colormap(depth_image: np.ndarray) -> np.ndarray:
        """Создание цветовой карты глубины"""
        return cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=Config.DEPTH_SCALE), 
            cv2.COLORMAP_JET
        )
