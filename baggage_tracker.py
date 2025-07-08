import numpy as np
import cv2
from collections import defaultdict
import uuid

class BaggageTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.tracked_objects = {}
        self.objects_to_finalize = []
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        
        # Новые параметры для устойчивости
        self.min_iou = 0.1  # Минимальный IoU для сопоставления
        self.area_threshold = 0.5  # Максимальное изменение площади
        

    def update(self, detections):
        """Обновление трекера с новыми детекциями"""
        # Предсказываем позиции на основе скорости
        self._predict_positions()
        
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.tracked_objects
        
        # Фильтруем детекции с валидными центроидами
        valid_detections = [d for d in detections if d.get('centroid') is not None]
        if not valid_detections:
            return self.tracked_objects
            
        input_centroids = np.array([d['centroid'] for d in valid_detections])
        
        if len(self.tracked_objects) == 0:
            for detection in valid_detections:
                self._register(detection)
        else:
            # Используем комбинированную метрику для сопоставления
            cost_matrix = self._compute_cost_matrix(valid_detections)
            
            if cost_matrix.size > 0:
                # Жадный алгоритм сопоставления
                assignments = self._greedy_assignment(cost_matrix)
                
                used_detections = set()
                used_objects = set()
                
                for obj_idx, det_idx in assignments:
                    obj_id = list(self.tracked_objects.keys())[obj_idx]
                    detection = valid_detections[det_idx]
                    
                    # Дополнительная проверка валидности сопоставления
                    if self._is_valid_match(obj_id, detection):
                        self._update_object(obj_id, detection)
                        used_objects.add(obj_idx)
                        used_detections.add(det_idx)
                
                # Обрабатываем неиспользованные объекты
                object_ids = list(self.tracked_objects.keys())
                for i, obj_id in enumerate(object_ids):
                    if i not in used_objects:
                        self.disappeared[obj_id] += 1
                        if self.disappeared[obj_id] > self.max_disappeared:
                            self._deregister(obj_id)
                
                # Регистрируем новые детекции
                for i, detection in enumerate(valid_detections):
                    if i not in used_detections:
                        self._register(detection)
        
        return self.tracked_objects
    
    def _predict_positions(self):
        """Предсказание позиций на основе скорости"""
        for obj in self.tracked_objects.values():
            prediction = obj.kalman.predict()
            obj.predicted_centroid = np.array([prediction[0,0], prediction[1,0]])
    
    def _compute_cost_matrix(self, detections):
        """Вычисление матрицы стоимости с учетом нескольких метрик"""
        object_ids = list(self.tracked_objects.keys())
        n_objects = len(object_ids)
        n_detections = len(detections)
        
        if n_objects == 0 or n_detections == 0:
            return np.array([])
        
        cost_matrix = np.zeros((n_objects, n_detections))
        
        for i, obj_id in enumerate(object_ids):
            obj = self.tracked_objects[obj_id]
            
            for j, det in enumerate(detections):
                # 1. Расстояние до предсказанной позиции
                pred_pos = getattr(obj, 'predicted_centroid', obj.centroid)
                dist_cost = np.linalg.norm(pred_pos - det['centroid'])
                
                # 2. IoU между bbox (если есть)
                iou_cost = 1.0
                if 'bbox' in det and hasattr(obj, 'bbox') and obj.bbox is not None:
                    iou = self._compute_iou(obj.bbox, det['bbox'])
                    iou_cost = 1 - iou
                
                # 3. Разница в площади
                area_cost = 0
                if 'area' in det and hasattr(obj, 'area') and obj.area > 0:
                    area_ratio = abs(obj.area - det['area']) / obj.area
                    area_cost = min(area_ratio, 1.0)
                
                # 4. Разница в классах (если разные классы - большая стоимость)
                class_cost = 0
                if 'class_id' in det and hasattr(obj, 'class_id'):
                    class_cost = 0 if obj.class_id == det['class_id'] else 0.5
                
                # Комбинированная стоимость
                cost_matrix[i, j] = (
                    0.4 * dist_cost +
                    0.3 * iou_cost * self.max_distance +
                    0.2 * area_cost * self.max_distance +
                    0.1 * class_cost * self.max_distance
                )
        
        return cost_matrix
    
    def _greedy_assignment(self, cost_matrix):
        """Жадный алгоритм сопоставления"""
        assignments = []
        used_rows = set()
        used_cols = set()
        
        # Находим минимальные стоимости
        flat_indices = np.argsort(cost_matrix.flatten())
        
        for flat_idx in flat_indices:
            row = flat_idx // cost_matrix.shape[1]
            col = flat_idx % cost_matrix.shape[1]
            
            if row in used_rows or col in used_cols:
                continue
            
            # Проверяем порог
            if cost_matrix[row, col] > self.max_distance:
                break
            
            assignments.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
        
        return assignments
    
    def _is_valid_match(self, obj_id, detection):
        """Дополнительная проверка валидности сопоставления"""
        obj = self.tracked_objects[obj_id]
        
        # Проверка IoU если есть bbox
        if 'bbox' in detection and hasattr(obj, 'bbox') and obj.bbox is not None:
            iou = self._compute_iou(obj.bbox, detection['bbox'])
            if iou < self.min_iou:
                print(f"[FILTER] IoU too low: {iou:.2f}")
                return False
        
        # Проверка изменения площади
        if 'area' in detection and hasattr(obj, 'area') and obj.area > 0:
            area_change = abs(obj.area - detection['area']) / obj.area
            if area_change > self.area_threshold:
                print(f"[FILTER] Area change too large: {area_change:.2f}")
                return False
        
        return True
    
    def _compute_iou(self, bbox1, bbox2):
        """Вычисление IoU между двумя bbox"""
        if bbox1 is None or bbox2 is None:
            return 0
        
        # Преобразуем к формату [x1, y1, x2, y2]
        if len(bbox1) == 4 and len(bbox2) == 4:
            if bbox1[2] < bbox1[0] or bbox1[3] < bbox1[1]:  # format [x, y, w, h]
                x1, y1, w1, h1 = bbox1
                bbox1 = [x1, y1, x1 + w1, y1 + h1]
            if bbox2[2] < bbox2[0] or bbox2[3] < bbox2[1]:  # format [x, y, w, h]
                x2, y2, w2, h2 = bbox2
                bbox2 = [x2, y2, x2 + w2, y2 + h2]
        
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Пересечение
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _register(self, detection):
        """Регистрация нового объекта"""
        self.tracked_objects[self.next_id] = TrackedObject(self.next_id, detection)
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _deregister(self, obj_id):
        """Подготовка объекта к удалению"""
        if obj_id in self.tracked_objects:
            # Сохраняем объект для финальной обработки
            self.objects_to_finalize.append(self.tracked_objects[obj_id])
            # Теперь удаляем
            del self.tracked_objects[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]

    def get_and_clear_finalized_objects(self):
        """Получить и очистить список объектов для финализации"""
        objects = self.objects_to_finalize.copy()
        self.objects_to_finalize.clear()
        return objects                   
    
    def _update_object(self, obj_id, detection):
        """Обновление существующего объекта"""
        obj = self.tracked_objects[obj_id]
        
        # Вычисляем скорость
        #f obj.centroid is not None:
        #    velocity = detection['centroid'] - obj.centroid
        #    # Сглаживаем скорость
        #    if hasattr(obj, 'velocity') and obj.velocity is not None:
        #        obj.velocity = 0.7 * obj.velocity + 0.3 * velocity
        #    else:
        #        obj.velocity = velocity
        
        obj.update(detection)
        self.disappeared[obj_id] = 0


class TrackedObject:
    """Класс для хранения информации об отслеживаемом объекте"""
    def __init__(self, obj_id, detection):
        self.id = obj_id
        self.unique_id = str(uuid.uuid4())
        self.centroid = np.array(detection['centroid']) if detection['centroid'] is not None else None
        self.bbox = detection.get('bbox')
        self.mask = detection.get('mask')
        self.class_id = detection.get('class_id')
        self.confidence = detection.get('confidence', 0.0)
        self.area = detection.get('area', 0)
        self._init_kalman(detection['centroid'])
        
        # Параметры движения
        self.velocity = None
        self.predicted_centroid = None
        
        # История для восстановления 3D
        self.rgb_crops = []
        
        # Параметры для 3D реконструкции
        self.depth_masks = []  # Список кортежей (mask, depth_values, timestamp)
        self.selected_masks = []  # Отобранные маски для реконструкции
        self.trigger_final_processing = False
        self.point_cloud = None
        self.mesh = None
        
    def _init_kalman(self, centroid):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()

    def update(self, detection):
        """Обновление параметров объекта"""
        if detection.get('centroid') is not None:
            self.centroid = np.array(detection['centroid'])
            measurement = np.array([[np.float32(detection['centroid'][0])],
                                    [np.float32(detection['centroid'][1])]])
            self.kalman.correct(measurement)
        if 'bbox' in detection:
            self.bbox = detection['bbox']
        if 'mask' in detection:
            self.mask = detection['mask']
        if 'confidence' in detection:
            self.confidence = detection['confidence']
        if 'area' in detection:
            self.area = detection['area']

    def add_depth_observation(self, mask, depth_values,rgb_values, timestamp):
        """Добавление наблюдения depth маски"""
        self.depth_masks.append({
            'mask': mask.copy(),
            'depth_values': depth_values.copy(),
            'rgb_values' : rgb_values.copy(),
            'timestamp': timestamp,
            'centroid': self.centroid.copy() if self.centroid is not None else None
        })