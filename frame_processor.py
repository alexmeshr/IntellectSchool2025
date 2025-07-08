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
        self.reconstructor = None
        if camera_intrinsics is not None:
            self.reconstructor = PointCloudReconstructor(camera_intrinsics)
        
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
                    rgb_values = color_image[tracked_obj.mask > 0]
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
        
        # Отбираем максимально разные маски
        selected_indices = self._select_diverse_masks(tracked_obj.depth_masks)
        tracked_obj.selected_masks = [tracked_obj.depth_masks[i] for i in selected_indices]
        
        # Восстанавливаем облако точек
        if self.reconstructor:
            # Используем ICP для лучшего выравнивания
            point_cloud = self.reconstructor.reconstruct_from_masks(
                tracked_obj.selected_masks, 
                with_icp=True
            )
            
            # Опционально создаем mesh
            mesh = None
            if len(point_cloud) > 1000:  # Достаточно точек для меша
                try:
                    mesh = self.reconstructor.create_mesh(point_cloud)
                except:
                    print(f"Object {tracked_obj.id}: не удалось создать mesh")
            
            # Сохраняем результат
            self.completed_objects.append({
                'id': tracked_obj.id,
                'unique_id': tracked_obj.unique_id,
                'point_cloud': point_cloud,
                'mesh': mesh,
                'rgb_images': tracked_obj.rgb_crops,
                'all_masks': tracked_obj.depth_masks,
                'selected_masks': tracked_obj.selected_masks,
                'timestamp': datetime.now()
            })
            

            print(f"Object {tracked_obj.id}: создано облако из {len(point_cloud)} точек на основании {len(tracked_obj.depth_masks)} масок")
    
    def _select_diverse_masks(self, depth_masks, min_masks=Config.MIN_OBSERVATIONS, max_masks=Config.MAX_OBSERVATIONS):
        """Отбор максимально разных масок"""
        n_masks = len(depth_masks)
        if n_masks <= min_masks:
            return list(range(n_masks))
        
        # Используем позиции центроидов для оценки разнообразия
        selected = [0, n_masks - 1]  # Первая и последняя
        
        # Добавляем маски с максимальным расстоянием до уже выбранных
        while len(selected) < min(max_masks, n_masks):
            max_dist = -1
            best_idx = -1
            
            for i in range(n_masks):
                if i in selected:
                    continue
                
                # Минимальное расстояние до выбранных масок
                min_dist_to_selected = float('inf')
                for j in selected:
                    if depth_masks[i]['centroid'] is not None and depth_masks[j]['centroid'] is not None:
                        dist = np.linalg.norm(
                            depth_masks[i]['centroid'] - depth_masks[j]['centroid']
                        )
                        min_dist_to_selected = min(min_dist_to_selected, dist)
                
                if min_dist_to_selected > max_dist:
                    max_dist = min_dist_to_selected
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(best_idx)
            else:
                break
        
        return sorted(selected)
    
    def _reconstruct_from_masks(self, tracked_obj):
        """Восстановление облака точек из выбранных масок"""
        all_points = []
        
        for mask_data in tracked_obj.selected_masks:
            mask = mask_data['mask']
            depth_values = mask_data['depth_values']
            
            # Получаем координаты пикселей маски
            y_coords, x_coords = np.where(mask > 0)
            
            if len(x_coords) == 0:
                continue
            
            # Конвертируем в 3D точки
            z = depth_values * self.camera_intrinsics['depth_scale']
            x = (x_coords - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx']
            y = (y_coords - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy']
            
            points = np.stack([x, y, z], axis=1)
            valid = z > 0
            all_points.extend(points[valid])
        
        return np.array(all_points)
    
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


