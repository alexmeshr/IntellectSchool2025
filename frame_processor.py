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

class FrameProcessor:
    def __init__(self, camera_intrinsics=None):
        self.detector = BaggageDetector()
        self.depth_processor = DepthProcessor()
        self.tracker = BaggageTracker(
            max_disappeared=Config.MAX_DISAPPEARED,
            max_distance=Config.MAX_DISTANCE
        )
        
        # Инициализируем реконструктор только если переданы параметры камеры
        self.reconstructor = None
        if camera_intrinsics is not None:
            self.reconstructor = PointCloudReconstructor(camera_intrinsics)
        
        self.frame_count = 0
        self.enable_tracking = True  # Флаг для включения/выключения трекинга
        
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
            tracked_objects = self.tracker.update(detection_results['baggage_detections'])
            
            # Сохранение наблюдений для 3D реконструкции
            if self.reconstructor is not None:
                self._collect_observations(
                    color_image, depth_image, tracked_objects, 
                    timestamp, camera_pose
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
            
            # Bbox
            x1, y1, x2, y2 = tracked_obj.bbox.astype(int)
            cv2.rectangle(color_with_mask, (x1, y1), (x2, y2), color, 2)
            
            # Подпись с ID и статусом
            label = f"ID:{obj_id}"
            if self.reconstructor and tracked_obj.point_cloud is not None:
                label += f" [3D:{len(tracked_obj.point_cloud.points)}pts]"
            elif hasattr(tracked_obj, 'rgb_crops'):
                label += f" [{len(tracked_obj.rgb_crops)}obs]"
            
            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(color_with_mask, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1 - 5), 
                         color, -1)
            cv2.putText(color_with_mask, label, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Центроид
            if tracked_obj.centroid:
                cv2.circle(color_with_mask, tracked_obj.centroid, 4, color, -1)
                cv2.circle(color_with_mask, tracked_obj.centroid, 5, (255, 255, 255), 1)
        
        return color_with_mask
    
    def _get_color_for_id(self, obj_id):
        """Генерация уникального цвета для ID"""
        # Используем HSV для ярких различимых цветов
        hue = (obj_id * 45) % 180  # Распределяем по цветовому кругу
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)
    
    def _collect_observations(self, color_image, depth_image, tracked_objects, 
                            timestamp, camera_pose):
        """Сбор наблюдений для 3D реконструкции"""
        for obj_id, tracked_obj in tracked_objects.items():
            # Вырезаем область багажа
            x1, y1, x2, y2 = tracked_obj.bbox.astype(int)
            
            # Проверяем границы
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(color_image.shape[1], x2), min(color_image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:  # Проверка валидности области
                rgb_crop = color_image[y1:y2, x1:x2]
                depth_crop = depth_image[y1:y2, x1:x2]
                mask_crop = tracked_obj.mask[y1:y2, x1:x2]
                
                # Добавляем наблюдение
                if len(tracked_obj.rgb_crops) < Config.MAX_OBSERVATIONS:
                    tracked_obj.add_observation(
                        rgb_crop, depth_crop, mask_crop, 
                        timestamp, camera_pose
                    )
                
                # Попытка реконструкции если достаточно наблюдений
                if len(tracked_obj.rgb_crops) >= Config.MIN_OBSERVATIONS:
                    if tracked_obj.point_cloud is None or self.frame_count % 30 == 0:
                        self.reconstructor.reconstruct_from_observations(tracked_obj)
    
    def set_tracking_enabled(self, enabled):
        """Включение/выключение трекинга"""
        self.enable_tracking = enabled
        if not enabled:
            # Очищаем трекер при выключении
            self.tracker = BaggageTracker(
                max_disappeared=Config.MAX_DISAPPEARED,
                max_distance=Config.MAX_DISTANCE
            )
    
    def export_point_clouds(self):
        """Экспорт всех облаков точек"""
        if not self.reconstructor:
            return []
            
        exported = []
        for obj_id, tracked_obj in self.tracker.tracked_objects.items():
            if tracked_obj.point_cloud is not None:
                filename = f"{Config.POINT_CLOUDS_DIR}/baggage_{tracked_obj.unique_id}.ply"
                self.reconstructor.save_point_cloud(tracked_obj, filename)
                
                # Попробуем создать mesh
                mesh = self.reconstructor.create_mesh(tracked_obj.point_cloud)
                mesh_filename = None
                if mesh is not None:
                    mesh_filename = f"{Config.POINT_CLOUDS_DIR}/baggage_{tracked_obj.unique_id}_mesh.ply"
                    self.reconstructor.save_mesh(mesh, mesh_filename)
                
                exported.append({
                    'id': obj_id,
                    'unique_id': tracked_obj.unique_id,
                    'pointcloud_file': filename,
                    'mesh_file': mesh_filename,
                    'points': len(tracked_obj.point_cloud.points),
                    'observations': len(tracked_obj.rgb_crops)
                })
        return exported

