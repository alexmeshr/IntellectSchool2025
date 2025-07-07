"""Модуль для восстановления облака точек багажа"""
import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple

class PointCloudReconstructor:
    def __init__(self, camera_intrinsics):
        """
        camera_intrinsics: dict с параметрами камеры
        {
            'fx': focal_x,
            'fy': focal_y,
            'cx': center_x,
            'cy': center_y,
            'depth_scale': scale
        }
        """
        self.intrinsics = camera_intrinsics
        
    def reconstruct_from_observations(self, tracked_object):
        """Восстановление облака точек из наблюдений объекта"""
        if len(tracked_object.rgb_crops) < 3:
            return None
        
        all_points = []
        all_colors = []
        
        for i, (rgb, depth, mask) in enumerate(zip(
            tracked_object.rgb_crops,
            tracked_object.depth_crops,
            tracked_object.masks_history
        )):
            # Конвертируем в облако точек
            points, colors = self._depth_to_pointcloud(rgb, depth, mask)
            
            # Если есть информация о позе камеры - трансформируем
            if i < len(tracked_object.camera_poses):
                points = self._transform_points(points, tracked_object.camera_poses[i])
            
            all_points.append(points)
            all_colors.append(colors)
        
        # Объединяем все точки
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Создаем облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Фильтрация выбросов
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Даунсэмплинг для оптимизации
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        tracked_object.point_cloud = pcd
        return pcd
    
    def _depth_to_pointcloud(self, rgb_image, depth_image, mask):
        """Конвертация depth изображения в облако точек"""
        h, w = depth_image.shape
        
        # Создаем сетку координат
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Применяем маску
        valid = (depth_image > 0) & (mask > 0)
        
        # Вычисляем 3D координаты
        z = depth_image[valid] * self.intrinsics['depth_scale']
        x = (xx[valid] - self.intrinsics['cx']) * z / self.intrinsics['fx']
        y = (yy[valid] - self.intrinsics['cy']) * z / self.intrinsics['fy']
        
        points = np.stack([x, y, z], axis=1)
        
        # Получаем цвета
        colors = rgb_image[valid] / 255.0
        
        return points, colors
    
    def _transform_points(self, points, pose):
        """Применение трансформации к точкам"""
        # pose должна быть матрицей 4x4
        if pose is None:
            return points
        
        # Добавляем гомогенную координату
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Применяем трансформацию
        transformed = (pose @ points_h.T).T
        
        return transformed[:, :3]
    
    def create_mesh(self, point_cloud, method='poisson'):
        """Создание меша из облака точек"""
        # Оценка нормалей
        point_cloud.estimate_normals()
        
        if method == 'poisson':
            # Poisson surface reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=9
            )
        else:
            # Ball pivoting
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, o3d.utility.DoubleVector(radii)
            )
        
        return mesh
    
    def save_point_cloud(self, tracked_object, filename):
        """Сохранение облака точек"""
        if tracked_object.point_cloud is not None:
            o3d.io.write_point_cloud(filename, tracked_object.point_cloud)
    
    def save_mesh(self, mesh, filename):
        """Сохранение меша"""
        if mesh is not None:
            o3d.io.write_triangle_mesh(filename, mesh)

