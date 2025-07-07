"""Модуль для детекции и сегментации людей"""
import numpy as np
import cv2
from ultralytics import YOLO
import logging
from interfaces import IPersonDetector
from config import Config

# Отключение логов YOLO
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class YOLOPersonDetector(IPersonDetector):
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Config.YOLO_MODEL
        
        # Загрузка модели с подавлением вывода
        self.yolo = YOLO(model_path, verbose=False)
        
    def detect_and_segment(self, image: np.ndarray) -> tuple:
        """Детекция и сегментация людей с сохранением багажа"""
        try:
            # Детекция всех объектов
            all_classes = [Config.PERSON_CLASS] + Config.BAGGAGE_CLASSES
            results = self.yolo(image, classes=all_classes, conf=Config.YOLO_CONFIDENCE, verbose=False)
            
            # Создаем маски
            person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            baggage_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            baggage_boxes = []
            
            # Обрабатываем результаты
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    for seg_mask, box, cls in zip(result.masks.data, result.boxes.xyxy, result.boxes.cls):
                        seg_mask_np = seg_mask.cpu().numpy()
                        seg_mask_resized = cv2.resize(seg_mask_np, (image.shape[1], image.shape[0]))
                        
                        if cls == Config.PERSON_CLASS:
                            person_mask[seg_mask_resized > 0.5] = 255
                        else:
                            baggage_mask[seg_mask_resized > 0.5] = 255
                            baggage_boxes.append(box.cpu().numpy())
            
            # Финальная маска: человек без багажа
            final_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(baggage_mask))
            
            # Буферная зона вокруг багажа
            if len(baggage_boxes) > 0 and np.any(baggage_mask):
                kernel = np.ones((10, 10), np.uint8)
                baggage_buffer = cv2.dilate(baggage_mask, kernel, iterations=1)
                final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(baggage_buffer))
            
            return final_mask, baggage_boxes
            
        except Exception as e:
            print(f"Ошибка сегментации: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), []

