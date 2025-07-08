# point_cloud_reconstructor.py
"""Модуль для восстановления облака точек багажа с использованием Open3D"""
import numpy as np
import cv2
import open3d as o3d
from typing import List, Dict

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
        
        # Создаем объект intrinsics для Open3D
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.o3d_intrinsics.set_intrinsics(
            width=640,  # Предполагаем стандартное разрешение
            height=480,
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy']
        )
        
    def reconstruct_from_masks(self, selected_masks: List[Dict], with_icp=True) -> np.ndarray:
        """Восстановление облака точек из масок глубины с выравниванием"""
        point_clouds = []
        
        # Создаем отдельные облака точек для каждой маски
        for i, mask_data in enumerate(selected_masks):
            mask = mask_data['mask']
            depth_values = mask_data['depth_values']
            
            # Создаем полное изображение глубины из маски
            depth_image = np.zeros_like(mask, dtype=np.float32)
            y_coords, x_coords = np.where(mask > 0)
            
            if len(x_coords) == 0:
                continue
                
            # Восстанавливаем изображение глубины
            depth_image[y_coords, x_coords] = depth_values
            
            # Конвертируем в формат Open3D (миллиметры в метры)
            depth_o3d = o3d.geometry.Image((depth_image).astype(np.uint16))
            
            # Создаем облако точек
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                self.o3d_intrinsics,
                depth_scale=1000.0  # Open3D ожидает depth в миллиметрах
            )
            
            # Удаляем нулевые точки
            points = np.asarray(pcd.points)
            valid = (points[:, 2] > 0) & (points[:, 2] < 5.0)
            pcd = pcd.select_by_index(np.where(valid)[0])
            
            if len(pcd.points) > 100:  # Минимум точек для облака
                point_clouds.append(pcd)
        
        if not point_clouds:
            return np.array([])
        
        # Объединяем облака точек с выравниванием
        if len(point_clouds) == 1:
            merged = point_clouds[0]
        else:
            merged = self._merge_point_clouds(point_clouds, with_icp)
        
        # Постобработка
        merged = self._postprocess_point_cloud(merged)
        
        return np.asarray(merged.points)
    
    def _merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud], 
                           with_icp: bool) -> o3d.geometry.PointCloud:
        """Объединение облаков точек с опциональным ICP выравниванием"""
        if not with_icp or len(point_clouds) < 2:
            # Простое объединение
            merged = o3d.geometry.PointCloud()
            for pcd in point_clouds:
                merged += pcd
            return merged
        
        # Выравнивание с помощью ICP
        merged = point_clouds[0]
        
        for i in range(1, len(point_clouds)):
            source = point_clouds[i]
            
            # Оценка нормалей для ICP
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            merged.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # ICP выравнивание
            threshold = 0.05  # 5см порог
            trans_init = np.eye(4)  # Начальная трансформация
            
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, merged, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            # Применяем трансформацию если выравнивание успешно
            if reg_p2p.fitness > 0.3:  # Минимум 30% перекрытия
                source.transform(reg_p2p.transformation)
            
            # Объединяем
            merged += source
        
        return merged
    
    def _postprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Постобработка облака точек"""
        # Удаление выбросов
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Даунсэмплинг для уменьшения шума
        pcd = pcd.voxel_down_sample(voxel_size=0.003)  # 3мм вокселы
        
        # Удаление изолированных точек
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.01)
        
        return pcd
    
    def create_mesh(self, points: np.ndarray, method='poisson') -> o3d.geometry.TriangleMesh:
        """Создание меша из массива точек"""
        if len(points) < 100:
            return None
            
        # Создаем облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Оценка нормалей
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        
        # Ориентация нормалей
        pcd.orient_normals_consistent_tangent_plane(30)
        
        if method == 'poisson':
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )
            
            # Обрезаем артефакты
            bbox = pcd.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)
        else:
            # Ball pivoting
            radii = [0.005, 0.01, 0.02]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        
        # Удаляем вырожденные треугольники
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        
        return mesh
    
    def save_point_cloud(self, points: np.ndarray, filename: str, with_normals=False):
        """Сохранение облака точек"""
        if len(points) == 0:
            return
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if with_normals:
            pcd.estimate_normals()
            
        o3d.io.write_point_cloud(filename, pcd)
    
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, filename: str):
        """Сохранение меша"""
        if mesh is not None:
            o3d.io.write_triangle_mesh(filename, mesh)
    
    def visualize_point_cloud(self, points: np.ndarray, window_name="Point Cloud"):
        """Быстрая визуализация облака точек"""
        if len(points) == 0:
            print("Нет точек для визуализации")
            return
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Раскрашиваем по высоте
        points_array = np.asarray(pcd.points)
        colors = plt.get_cmap('viridis')((points_array[:, 2] - points_array[:, 2].min()) / 
                                         (points_array[:, 2].max() - points_array[:, 2].min()))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([pcd], window_name=window_name)