# dimension_estimator.py
"""Модуль для оценки габаритов объекта по лучшему кадру"""
import numpy as np
import cv2
import open3d as o3d

class DimensionEstimator:
    def __init__(self, camera_intrinsics):
        self.intrinsics = camera_intrinsics
        
    def select_best_frame(self, depth_masks):
        """Выбор лучшего кадра по критериям полноты и качества"""
        best_score = -1
        best_idx = -1
        scores_debug = []
        
        for i, mask_data in enumerate(depth_masks):
            mask = mask_data['mask']
            depth_values = mask_data['depth_values']
            
            # Базовая проверка
            valid_depth = depth_values[depth_values > 0]
            if len(valid_depth) < 100:
                continue
            
            # 1. Полнота данных (процент валидных пикселей в маске)
            completeness = len(valid_depth) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
            
            # 2. Размер проекции (нормализованный по среднему)
            area = np.sum(mask > 0)
            avg_area = np.mean([np.sum(m['mask'] > 0) for m in depth_masks])
            size_score = min(area / avg_area, 2.0) if avg_area > 0 else 1.0
            
            # 3. Качество глубины (исключаем выбросы)
            q25, q75 = np.percentile(valid_depth, [25, 75])
            iqr = q75 - q25
            inliers = valid_depth[(valid_depth >= q25 - 1.5*iqr) & (valid_depth <= q75 + 1.5*iqr)]
            depth_quality = len(inliers) / len(valid_depth) if len(valid_depth) > 0 else 0
            
            # 4. Объемность - разброс глубин показывает 3D структуру
            if len(inliers) > 10:
                depth_range = np.ptp(inliers)  # peak-to-peak (max - min)
                depth_mean = np.mean(inliers)
                # Нормализуем разброс относительно средней глубины
                volume_score = min(depth_range / (depth_mean * 0.3), 1.0)  # ожидаем ~30% вариации
            else:
                volume_score = 0
            
            # 5. Аспектное соотношение - избегаем слишком вытянутых проекций
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                width = np.ptp(x_coords)
                height = np.ptp(y_coords)
                aspect = min(width, height) / max(width, height) if max(width, height) > 0 else 0
            else:
                aspect = 0
            
            # Взвешенный скор с приоритетом на 3D видимость
            score = (completeness * 0.2 +
                    size_score * 0.3 +
                    depth_quality * 0.2 +
                    volume_score * 0.2 +  # Важнейший критерий
                    aspect * 0.1)
            
            scores_debug.append({
                'frame': i,
                'score': score,
                'volume': volume_score,
                'quality': depth_quality
            })
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Debug вывод топ-3 кадров
        if len(scores_debug) > 0:
            sorted_scores = sorted(scores_debug, key=lambda x: x['score'], reverse=True)[:3]
            print(f"Топ кадры: ", [(s['frame'], f"score:{s['score']:.3f}, vol:{s['volume']:.3f}") for s in sorted_scores])
                
        return best_idx if best_idx >= 0 else 0
    
    def mask_to_point_cloud(self, mask, depth_image, rgb_image=None):
        """Преобразование маски глубины в облако точек"""
        # Получаем координаты пикселей под маской
        y_coords, x_coords = np.where(mask > 0)
        depths = depth_image[y_coords, x_coords]
        
        # Фильтруем невалидные значения
        valid = depths > 0
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depths = depths[valid]
        
        if len(depths) == 0:
            return np.array([])
        
        # Преобразование в 3D координаты
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        
        z = depths / 1000.0  # мм в метры
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)
        
        # Добавляем цвета если есть
        if rgb_image is not None:
            colors = rgb_image[y_coords, x_coords]
            return points, colors
            
        return points
    
    def remove_outliers(self, points, nb_neighbors=20, std_ratio=2.0):
        """Удаление выбросов из облака точек"""
        if len(points) < nb_neighbors:
            return points
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Статистическое удаление выбросов
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        return np.asarray(pcd_clean.points)
    
    def compute_oriented_bbox(self, points):
        """Вычисление ориентированного bounding box через Open3D"""
        if len(points) < 10:
            return None
            
        # Создаем облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Вычисляем ориентированный bbox
        obb = pcd.get_oriented_bounding_box()
        
        # Извлекаем параметры
        dimensions = np.array(obb.extent)
        
        # Сортируем размеры по убыванию
        sorted_indices = np.argsort(dimensions)[::-1]
        sorted_dims = dimensions[sorted_indices]
        
        # Сортируем столбцы матрицы поворота соответственно
        rotation_matrix = np.array(obb.R)
        sorted_rotation = rotation_matrix[:, sorted_indices]
        
        return {
            'dimensions': sorted_dims,  # [длина, ширина, высота] в метрах
            'centroid': np.array(obb.center),  # Центр bbox
            'rotation_matrix': sorted_rotation,  # Матрица поворота с отсортированными осями
            'volume': np.prod(sorted_dims)
        }
    
    def estimate_dimensions(self, depth_masks):
        """Основной метод оценки габаритов"""
        if len(depth_masks) == 0:
            return None
            
        # Выбираем лучший кадр
        best_idx = self.select_best_frame(depth_masks)
        best_mask_data = depth_masks[best_idx]
        
        # Получаем полное изображение глубины для выбранного кадра
        mask = best_mask_data['mask']
        depth_values = best_mask_data['depth_values']
        
        # Восстанавливаем полное изображение глубины из маски
        depth_image = np.zeros_like(mask, dtype=np.float32)
        depth_image[mask > 0] = depth_values
        
        # Преобразуем в облако точек
        points = self.mask_to_point_cloud(mask, depth_image)
        
        if len(points) < 10:
            return None
            
        # Удаляем выбросы
        clean_points = self.remove_outliers(points)
        
        if len(clean_points) < 10:
            return None
            
        # Вычисляем ориентированный bbox
        bbox_data = self.compute_oriented_bbox(clean_points)
        
        if bbox_data is None:
            return None

        rgb = best_mask_data['rgb_values']
        return {
            'dimensions': bbox_data['dimensions'],
            'volume': bbox_data['volume'],
            'point_cloud': clean_points,
            'centroid': bbox_data['centroid'],
            'rotation_matrix': bbox_data['rotation_matrix'],
            'frame_index': best_idx,
            'num_points': len(clean_points),
            'rgb_image': rgb
        }