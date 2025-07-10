# dimension_estimator.py
"""Модуль для оценки габаритов объекта по лучшему кадру"""
import numpy as np
import cv2
import open3d as o3d
from config import Config
from scipy import stats


class DimensionEstimator:
    def __init__(self, camera_intrinsics):
        self.intrinsics = camera_intrinsics

    def select_best_frame(self, depth_masks):
        """Выбор лучшего кадра по критериям полноты и качества с проверкой на аномалии"""
        best_score = -1
        best_idx = -1
        scores_debug = []
        
        # Инициализируем кэш для избежания повторных вычислений
        self._frame_estimations_cache = {}
        
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
                    volume_score * 0.2 +  
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
        #if len(scores_debug) > 0:
        #    sorted_scores = sorted(scores_debug, key=lambda x: x['score'], reverse=True)[:3]
        #    print(f"Топ кадры: ", [(s['frame'], f"score:{s['score']:.3f}, vol:{s['volume']:.3f}") for s in sorted_scores])
        
        # Проверяем на аномалии если есть достаточно кадров
        if len(scores_debug) >= Config.TOP_N_ESTIMATIONS:
            validated_idx = self._validate_best_frame(depth_masks, scores_debug)
            if validated_idx is not None:
                return validated_idx
                
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
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)

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
            
        # Выбираем лучший кадр с проверкой на аномалии
        best_idx = self.select_best_frame(depth_masks)
        best_mask_data = depth_masks[best_idx]
        
        # Используем общий метод обработки
        result = self._process_mask_to_bbox(best_mask_data)
        if result is None:
            return None
            
        bbox_data, clean_points = result
        
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

    def _validate_best_frame(self, depth_masks, scores_debug):
        """Проверка лучшего кадра на аномалии относительно других топ-кадров"""
        top_n = Config.TOP_N_ESTIMATIONS

        # Сортируем кадры по скору и берем топ-N
        sorted_scores = sorted(
            scores_debug, key=lambda x: x['score'], reverse=True)
        top_frames = sorted_scores[:top_n]

        # Получаем оценки габаритов для топ-кадров
        estimations = []
        for frame_data in top_frames:
            frame_idx = frame_data['frame']
            estimation = self._get_frame_estimation(depth_masks[frame_idx])
            if estimation is not None:
                estimation['frame_idx'] = frame_idx
                estimation['score'] = frame_data['score']
                estimations.append(estimation)

        if len(estimations) < 2:
            # Недостаточно данных для сравнения
            return None

        # Анализируем аномалии
        outlier_threshold = Config.OUTLIER_THRESHOLD

        # Извлекаем габариты и объемы
        dimensions = np.array([est['dimensions'] for est in estimations])
        print("TOP DIMENSIONS:")
        for i, d in enumerate(dimensions):
            print(f"{i}: {d}")
        volumes = np.array([est['volume'] for est in estimations])

        # Вычисляем Z-скоры для лучшего кадра (первый в списке)
        best_estimation = estimations[0]

        # Z-скор по объему
        if len(volumes) > 1 and np.std(volumes) > 0:
            volume_z = np.abs(stats.zscore(volumes)[0])
        else:
            volume_z = 0

        # Z-скоры по габаритам
        dim_z_scores = []
        for dim_idx in range(3):
            dim_values = dimensions[:, dim_idx]
            if len(dim_values) > 1 and np.std(dim_values) > 0:
                z_score = np.abs(stats.zscore(dim_values)[0])
                dim_z_scores.append(z_score)

        max_dim_z = max(dim_z_scores) if dim_z_scores else 0

        # Проверяем на аномалию
        is_outlier = (volume_z > outlier_threshold or max_dim_z >
                      outlier_threshold)

        if is_outlier:
            print(f"Кадр {best_estimation['frame_idx']} признан аномальным: "
                  f"volume_z={volume_z:.2f}, max_dim_z={max_dim_z:.2f}")

            # Ищем лучший не-аномальный кадр
            for i, est in enumerate(estimations[1:], 1):
                # Пересчитываем Z-скоры без текущего кандидата
                # Удаляем лучший кадр
                test_dims = np.delete(dimensions, 0, axis=0)
                test_vols = np.delete(volumes, 0)

                # Добавляем текущий кандидат в начало
                test_dims = np.vstack([dimensions[i], test_dims])
                test_vols = np.hstack([volumes[i], test_vols])

                # Вычисляем Z-скоры для кандидата
                vol_z = np.abs(stats.zscore(test_vols)[0]) if len(
                    test_vols) > 1 and np.std(test_vols) > 0 else 0

                candidate_dim_z = []
                for dim_idx in range(3):
                    dim_vals = test_dims[:, dim_idx]
                    if len(dim_vals) > 1 and np.std(dim_vals) > 0:
                        z_score = np.abs(stats.zscore(dim_vals)[0])
                        candidate_dim_z.append(z_score)

                max_candidate_z = max(
                    candidate_dim_z) if candidate_dim_z else 0

                # Если кандидат не аномальный, выбираем его
                if vol_z <= outlier_threshold and max_candidate_z <= outlier_threshold:
                    print(f"Выбран кадр {est['frame_idx']} вместо аномального")
                    return est['frame_idx']

            # Если все кандидаты аномальные, используем второй лучший
            if len(estimations) > 1:
                print(
                    f"Все кандидаты аномальные, используем второй лучший: {estimations[1]['frame_idx']}")
                return estimations[1]['frame_idx']

        # Лучший кадр не аномальный
        return None
    

    def _process_mask_to_bbox(self, mask_data):
        """Общий метод для преобразования маски в bbox данные"""
        mask = mask_data['mask']
        depth_values = mask_data['depth_values']
        
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
            
        return bbox_data, clean_points
    
    def _get_frame_estimation(self, mask_data, frame_idx=None):
        """Получение оценки габаритов для одного кадра (вспомогательный метод)"""
        # Проверяем кэш
        if hasattr(self, '_frame_estimations_cache') and frame_idx is not None and frame_idx in self._frame_estimations_cache:
            return self._frame_estimations_cache[frame_idx]
        
        # Используем общий метод обработки
        result = self._process_mask_to_bbox(mask_data)
        if result is None:
            return None
            
        bbox_data, clean_points = result
        
        estimation = {
            'dimensions': bbox_data['dimensions'],
            'volume': bbox_data['volume'],
            'point_cloud': clean_points,
            'centroid': bbox_data['centroid'],
            'rotation_matrix': bbox_data['rotation_matrix'],
            'num_points': len(clean_points)
        }
        
        # Сохраняем в кэш
        if hasattr(self, '_frame_estimations_cache') and frame_idx is not None:
            self._frame_estimations_cache[frame_idx] = estimation
        
        return estimation
