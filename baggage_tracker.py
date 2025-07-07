import numpy as np
import cv2
from collections import defaultdict
import uuid

class BaggageTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.tracked_objects = {}  # ID -> TrackedObject
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        
    def update(self, detections):
        """Обновление трекера с новыми детекциями"""
        # Если нет детекций
        if len(detections) == 0:
            # Увеличиваем счетчик пропавших для всех объектов
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.tracked_objects
        
        # Получаем центроиды новых детекций
        input_centroids = np.array([d['centroid'] for d in detections if d['centroid'] is not None])
        
        # Если нет отслеживаемых объектов - регистрируем все новые
        if len(self.tracked_objects) == 0:
            for i, detection in enumerate(detections):
                self._register(detection)
        else:
            # Сопоставляем существующие объекты с новыми детекциями
            object_ids = list(self.tracked_objects.keys())
            tracked_centroids = [self.tracked_objects[obj_id].centroid for obj_id in object_ids]
            
            # Вычисляем матрицу расстояний
            distances = self._compute_distances(tracked_centroids, input_centroids)
            
            # Находим минимальные расстояния и сопоставляем объекты
            if distances.size > 0:
                # Сортируем по расстоянию
                sorted_indices = np.argsort(distances.flatten())
                used_detections = set()
                used_objects = set()
                
                for idx in sorted_indices:
                    row = idx // distances.shape[1]
                    col = idx % distances.shape[1]
                    
                    if row in used_objects or col in used_detections:
                        continue
                    
                    # Проверяем максимальное расстояние
                    if distances[row, col] > self.max_distance:
                        break
                    
                    obj_id = object_ids[row]
                    self._update_object(obj_id, detections[col])
                    used_objects.add(row)
                    used_detections.add(col)
                
                # Обрабатываем неиспользованные объекты
                unused_objects = set(range(len(object_ids))) - used_objects
                for row in unused_objects:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self._deregister(obj_id)
                
                # Регистрируем новые детекции
                unused_detections = set(range(len(detections))) - used_detections
                for col in unused_detections:
                    self._register(detections[col])
        
        return self.tracked_objects
    
    def _register(self, detection):
        """Регистрация нового объекта"""
        self.tracked_objects[self.next_id] = TrackedObject(self.next_id, detection)
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _deregister(self, obj_id):
        """Удаление объекта из трекинга"""
        del self.tracked_objects[obj_id]
        del self.disappeared[obj_id]
    
    def _update_object(self, obj_id, detection):
        """Обновление существующего объекта"""
        self.tracked_objects[obj_id].update(detection)
        self.disappeared[obj_id] = 0
    
    def _compute_distances(self, centroids1, centroids2):
        """Вычисление матрицы евклидовых расстояний"""
        centroids1 = np.array(centroids1)
        centroids2 = np.array(centroids2)
        
        if len(centroids1) == 0 or len(centroids2) == 0:
            return np.array([])
        
        # Вычисляем попарные расстояния
        distances = np.zeros((len(centroids1), len(centroids2)))
        for i, c1 in enumerate(centroids1):
            for j, c2 in enumerate(centroids2):
                distances[i, j] = np.linalg.norm(c1 - c2)
        
        return distances


class TrackedObject:
    """Класс для хранения информации об отслеживаемом объекте"""
    def __init__(self, obj_id, detection):
        self.id = obj_id
        self.unique_id = str(uuid.uuid4())  # Уникальный ID для облака точек
        self.centroid = detection['centroid']
        self.bbox = detection['bbox']
        self.mask = detection['mask']
        self.class_id = detection['class_id']
        self.confidence = detection['confidence']
        self.area = detection['area']
        
        # История для восстановления 3D
        self.rgb_crops = []  # Вырезанные изображения багажа
        self.depth_crops = []  # Соответствующие карты глубины
        self.masks_history = []  # История масок
        self.camera_poses = []  # Позы камеры (если доступны)
        self.timestamps = []
        
        # Параметры для 3D реконструкции
        self.point_cloud = None
        self.mesh = None
        
    def update(self, detection):
        """Обновление параметров объекта"""
        self.centroid = detection['centroid']
        self.bbox = detection['bbox']
        self.mask = detection['mask']
        self.confidence = detection['confidence']
        self.area = detection['area']
    
    def add_observation(self, rgb_crop, depth_crop, mask, timestamp, camera_pose=None):
        """Добавление наблюдения для 3D реконструкции"""
        self.rgb_crops.append(rgb_crop)
        self.depth_crops.append(depth_crop)
        self.masks_history.append(mask)
        self.timestamps.append(timestamp)
        if camera_pose is not None:
            self.camera_poses.append(camera_pose)