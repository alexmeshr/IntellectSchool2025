# frame_processor.py
"""Основной процессор кадров с поддержкой трекинга"""
import cv2
import numpy as np
from datetime import datetime
from baggage_detector import BaggageDetector
from baggage_tracker import BaggageTracker
from point_cloud_reconstructor import PointCloudReconstructor
from depth_processor import DepthProcessor
from config import Config
from dimension_estimator import DimensionEstimator

def non_max_suppression_bbox(detections, iou_threshold=0.5):
        boxes = [d['bbox'] for d in detections]
        scores = [d.get('confidence', 1.0) for d in detections]

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.1, nms_threshold=iou_threshold)
        indices = indices.flatten() if len(indices) > 0 else []
        return [detections[i] for i in indices]

class FrameProcessor:
    def __init__(self, camera_intrinsics=None):
        self.detector = BaggageDetector()
        self.depth_processor = DepthProcessor()
        self.tracker = BaggageTracker(
            max_disappeared=Config.MAX_DISAPPEARED,
            max_distance=Config.MAX_DISTANCE
        )
        
        # Инициализируем реконструктор только если переданы параметры камеры
        self.dimension_estimator = None
        if camera_intrinsics is not None:
            self.dimension_estimator = DimensionEstimator(camera_intrinsics)
        
        self.frame_count = 0
        self.enable_tracking = True  # Флаг для включения/выключения трекинга
        self.completed_objects = []
        

    def process_frame(self, color_image, depth_image, camera_pose=None):
        """Обработка кадра: детекция, сегментация, очистка depth"""
        self.frame_count += 1
        timestamp = datetime.now()
        
        # Детекция и сегментация
        detection_results = self.detector.detect_and_segment(color_image)
        person_mask = detection_results['person_mask']
        baggage_mask = detection_results['baggage_mask']
        
        # Очистка карты глубины - оставляем только багаж
        cleaned_depth = self.depth_processor.keep_only_baggage(depth_image, baggage_mask)
        
        # Создание визуализаций
        depth_colormap = self.depth_processor.create_colormap(depth_image)
        cleaned_depth_colormap = self.depth_processor.create_colormap(cleaned_depth)
        
        # Трекинг багажа (если включен)
        tracked_objects = {}
        if self.enable_tracking:
            # Сначала проверяем объекты на финальную обработку
            finalized_objects = self.tracker.get_and_clear_finalized_objects()
            # Обрабатываем объекты перед обновлением трекера
            for obj in finalized_objects:
                self._finalize_object(obj)
            
            # Обновляем трекер
            filtered_detections = non_max_suppression_bbox(detection_results['baggage_detections'])
            tracked_objects = self.tracker.update(filtered_detections)
            
            # Сохраняем depth маски для активных объектов
            for obj_id, tracked_obj in tracked_objects.items():
                if tracked_obj.mask is not None:
                    # Извлекаем значения глубины под маской
                    depth_values = depth_image[tracked_obj.mask > 0]
                    rgb_values = color_image#[tracked_obj.mask > 0]
                    if len(depth_values) > 0:
                        tracked_obj.add_depth_observation(
                            tracked_obj.mask, 
                            depth_values,
                            rgb_values,
                            timestamp
                        )
        
        
        # Визуализация масок на RGB
        if self.enable_tracking and tracked_objects:
            # Визуализация с трекингом
            color_with_mask = self._visualize_with_tracking(
                color_image, person_mask, tracked_objects
            )
        else:
            # Оригинальная визуализация без трекинга
            color_with_mask = self._visualize_detections(
                color_image, person_mask, baggage_mask
            )
        
        # Объединение изображений
        combined = np.hstack((color_with_mask, depth_colormap, cleaned_depth_colormap))
        
        return {
            'color_with_mask': color_with_mask,
            'depth_colormap': depth_colormap,
            'cleaned_depth_colormap': cleaned_depth_colormap,
            'combined': combined,
            'person_mask': person_mask,
            'baggage_mask': baggage_mask,
            'cleaned_depth': cleaned_depth,
            'tracked_objects': tracked_objects,
            'frame_count': self.frame_count
        }
    
    def _visualize_detections(self, color_image, person_mask, baggage_mask):
        """Оригинальная визуализация детекций на изображении"""
        color_with_mask = color_image.copy()
        
        # Полупрозрачная зеленая маска для людей, синяя для багажа
        overlay = color_image.copy()
        overlay[person_mask > 0] = [0, 255, 0]
        overlay[baggage_mask > 0] = [0, 0, 255]
        color_with_mask = cv2.addWeighted(color_image, 0.5, overlay, 0.3, 0)
        
        # Контуры людей (зеленые)
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_with_mask, contours, -1, (0, 255, 0), 2)
        
        # Контуры багажа (синие)
        contours, _ = cv2.findContours(baggage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_with_mask, contours, -1, (0, 0, 255), 2)
        
        return color_with_mask
    
    def _visualize_with_tracking(self, color_image, person_mask, tracked_objects):
        """Визуализация с трекингом багажа"""
        color_with_mask = color_image.copy()
        
        # Полупрозрачная зеленая маска для людей
        overlay = color_image.copy()
        overlay[person_mask > 0] = [0, 255, 0]
        
        # Применяем маску людей
        color_with_mask = cv2.addWeighted(color_image, 0.7, overlay, 0.3, 0)
        
        # Контуры людей (зеленые)
        person_contours, _ = cv2.findContours(
            person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(color_with_mask, person_contours, -1, (0, 255, 0), 2)
        
        # Отслеживаемый багаж с уникальными цветами для каждого ID
        for obj_id, tracked_obj in tracked_objects.items():
            # Получаем уникальный цвет для ID
            color = self._get_color_for_id(obj_id)
            
            # Полупрозрачная маска для багажа
            overlay = color_with_mask.copy()
            overlay[tracked_obj.mask > 0] = color
            color_with_mask = cv2.addWeighted(color_with_mask, 0.7, overlay, 0.3, 0)
            
            # Контур маски
            contours, _ = cv2.findContours(
                tracked_obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(color_with_mask, contours, -1, color, 2)
            
            # Центроид
            #if tracked_obj.centroid:
            #    cv2.circle(color_with_mask, tracked_obj.centroid, 4, color, -1)
            #    cv2.circle(color_with_mask, tracked_obj.centroid, 5, (255, 255, 255), 1)
        
        return color_with_mask
    
    def _get_color_for_id(self, obj_id):
        """Генерация уникального цвета для ID"""
        # Используем HSV для ярких различимых цветов
        hue = (obj_id * 45) % 180  # Распределяем по цветовому кругу
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)
    
    
    def set_tracking_enabled(self, enabled):
        """Включение/выключение трекинга"""
        self.enable_tracking = enabled
        if not enabled:
            # Очищаем трекер при выключении
            self.tracker = BaggageTracker(
                max_disappeared=Config.MAX_DISAPPEARED,
                max_distance=Config.MAX_DISTANCE
            )
    
    def _finalize_object(self, tracked_obj):
        """Финальная обработка объекта перед удалением"""
        if len(tracked_obj.depth_masks) < Config.MIN_OBSERVATIONS:
            print(f"Object {tracked_obj.id}: недостаточно масок ({len(tracked_obj.depth_masks)})")
            return
        
        # Используем DimensionEstimator для оценки габаритов
        if self.dimension_estimator:
            result = self.dimension_estimator.estimate_dimensions(tracked_obj.depth_masks)
            
            if result is None:
                print(f"Object {tracked_obj.id}: не удалось оценить габариты")
                return
            
            # Сохраняем результат
            self.completed_objects.append({
                'id': tracked_obj.id,
                'unique_id': tracked_obj.unique_id,
                'all_masks': tracked_obj.depth_masks,
                'dimensions': result['dimensions'],  # [длина, ширина, высота] в метрах
                'volume': result['volume'],
                'point_cloud': result['point_cloud'],
                'centroid': result['centroid'],
                'rotation_matrix': result['rotation_matrix'],
                'best_frame_index': result['frame_index'],
                'num_points': result['num_points'],
                'rgb_image': result['rgb_image'],
                'timestamp': datetime.now()
            })
            
            print(f"Object {tracked_obj.id}: габариты {result['dimensions'][0]:.3f}x{result['dimensions'][1]:.3f}x{result['dimensions'][2]:.3f} м, "
                f"объем {result['volume']:.4f} м³, {result['num_points']} точек")
        else:
            print(f"Object {tracked_obj.id}: DimensionEstimator не инициализирован")

    
    @property
    def camera_intrinsics(self):
        """Получение параметров камеры"""
        if self.reconstructor:
            return self.reconstructor.intrinsics
        return None
    
    def get_completed_objects(self):
        """Получение завершенных объектов с облаками точек"""
        return self.completed_objects
    
    def save_debug_data(self, output_dir="debug_masks"):
        """Сохранение данных для дебага"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, obj_data in enumerate(self.completed_objects):
            obj_dir = os.path.join(output_dir, f"object_{obj_data['id']}_{obj_data['unique_id'][:8]}")
            os.makedirs(obj_dir, exist_ok=True)
            
            # Сохраняем RGB изображение
            if obj_data['rgb_image'] is not None:
                cv2.imwrite(os.path.join(obj_dir, "rgb.jpg"), obj_data['rgb_image'])
            
            # Сохраняем выбранные маски
            for j, mask_data in enumerate(obj_data['selected_masks']):
                mask_vis = (mask_data['mask'] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(obj_dir, f"mask_{j}.png"), mask_vis)
            
            # Сохраняем облако точек
            if len(obj_data['point_cloud']) > 0:
                np.save(os.path.join(obj_dir, "point_cloud.npy"), obj_data['point_cloud'])


