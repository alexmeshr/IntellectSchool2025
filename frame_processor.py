"""Основной процессор кадров"""
import cv2
import numpy as np
from person_detector import YOLOPersonDetector
from depth_processor import DepthProcessor

class FrameProcessor:
    def __init__(self):
        self.person_detector = YOLOPersonDetector()
        self.depth_processor = DepthProcessor()
        
    def process_frame(self, color_image, depth_image):
        """Обработка кадра: детекция, сегментация, очистка depth"""
        # Детекция и сегментация
        person_mask, baggage_boxes = self.person_detector.detect_and_segment(color_image)
        
        # Очистка карты глубины
        cleaned_depth = self.depth_processor.remove_person_from_depth(depth_image, person_mask)
        
        # Создание визуализаций
        depth_colormap = self.depth_processor.create_colormap(depth_image)
        cleaned_depth_colormap = self.depth_processor.create_colormap(cleaned_depth)
        
        # Визуализация масок на RGB
        color_with_mask = self._visualize_detections(color_image, person_mask, baggage_boxes)
        
        # Объединение изображений
        combined = np.hstack((color_with_mask, depth_colormap, cleaned_depth_colormap))
        
        return {
            'color_with_mask': color_with_mask,
            'depth_colormap': depth_colormap,
            'cleaned_depth_colormap': cleaned_depth_colormap,
            'combined': combined,
            'person_mask': person_mask,
            'cleaned_depth': cleaned_depth
        }
    
    def _visualize_detections(self, color_image, person_mask, baggage_boxes):
        """Визуализация детекций на изображении"""
        color_with_mask = color_image.copy()
        
        # Полупрозрачная зеленая маска для людей
        overlay = color_image.copy()
        overlay[person_mask > 0] = [0, 255, 0]
        color_with_mask = cv2.addWeighted(color_image, 0.7, overlay, 0.3, 0)
        
        # Контуры людей
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_with_mask, contours, -1, (0, 255, 0), 2)
        
        # Bbox'ы багажа
        for box in baggage_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(color_with_mask, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(color_with_mask, "Baggage", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return color_with_mask
