"""Модуль для детекции и сегментации багажа с поддержкой трекинга"""
import numpy as np
import cv2
from ultralytics import YOLO
import logging
from config import Config

# Отключение логов YOLO
logging.getLogger('ultralytics').setLevel(logging.ERROR)
#проверить дисторсию, 

class BaggageDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Config.YOLO_MODEL

        # Загрузка модели с подавлением вывода
        self.yolo = YOLO(model_path, verbose=False)

    def detect_and_segment(self, image: np.ndarray) -> dict:
        """Детекция людей и багажа с полной информацией для трекинга"""
        try:
            # Детекция всех объектов
            all_classes =  Config.BAGGAGE_CLASSES #+ [Config.PERSON_CLASS] 
            results = self.yolo(image, classes=all_classes,
                                conf=Config.BAG_CONFIDENCE, verbose=False)

            # Создаем маски и списки для трекинга
            person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            baggage_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            baggage_detections = []
            person_boxes = []

            # Обрабатываем результаты
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    for seg_mask, box, cls, conf in zip(
                        result.masks.data,
                        result.boxes.xyxy,
                        result.boxes.cls,
                        result.boxes.conf
                    ):
                        seg_mask_np = seg_mask.cpu().numpy()
                        seg_mask_resized = cv2.resize(
                            seg_mask_np, (image.shape[1], image.shape[0]))

                        if cls == Config.PERSON_CLASS:
                            if conf >= Config.PERSON_CONFIDENCE:
                                person_mask[seg_mask_resized > 0.5] = 255
                                person_boxes.append(box.cpu().numpy())
                        else:
                            # Сохраняем детальную информацию о багаже
                            baggage_mask[seg_mask_resized > 0.5] = 255

                            # Создаем бинарную маску для этого объекта
                            individual_mask = (
                                seg_mask_resized > 0.5).astype(np.uint8) * 255

                            baggage_detections.append({
                                'bbox': box.cpu().numpy(),
                                'mask': individual_mask,
                                'class_id': int(cls),
                                'confidence': float(conf),
                                'centroid': self._calculate_centroid(individual_mask),
                                'area': np.sum(individual_mask > 0)
                            })

            # Убираем людей из маски багажа (на случай пересечений)
            #cleaned_baggage_mask = cv2.bitwise_and(
            #    baggage_mask, cv2.bitwise_not(person_mask))

            return {
                'person_mask': person_mask,
                'baggage_mask': baggage_mask,
                'baggage_detections': baggage_detections,
                'person_boxes': person_boxes
            }

        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return {
                'person_mask': np.zeros(image.shape[:2], dtype=np.uint8),
                'baggage_mask': np.zeros(image.shape[:2], dtype=np.uint8),
                'baggage_detections': [],
                'person_boxes': []
            }

    def _calculate_centroid(self, mask):
        """Вычисление центроида маски"""
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return None
