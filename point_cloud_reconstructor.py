# point_cloud_reconstructor.py
"""Модуль для восстановления облака точек багажа с использованием Open3D"""
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
import cv2
from scipy.spatial.distance import cdist
from config import Config
import pyrealsense2 as rs


class PointCloudReconstructor:
    def __init__(self, camera_intrinsics: Dict):
        """
        Улучшенная система восстановления объектов с использованием 
        современных методов поверхностной реконструкции
        """
        self.intrinsics = camera_intrinsics


        # Параметры для оптимизации
        self.depth_scale = 1000.0
        self.min_points_threshold = 50
        self.voxel_size = 0.002  # 2мм вокселы для более точной обработки

    def reconstruct_from_masks(self, selected_masks: List[Dict], with_icp=True,
                               reconstruction_method: str = 'poisson'):
        """
        Оптимизированное восстановление объекта с возможностью получения mesh

        Args:
            selected_masks: Список масок с depth данными
            method: 'poisson', 'ball_pivoting', 'alpha_shape', или 'delaunay'

        Returns:
            Tuple[point_cloud, mesh] - облако точек и mesh (если метод поддерживает)
        """
        # Этап 1: Быстрое создание облака точек без ICP
        point_clouds = self._create_point_clouds_vectorized(selected_masks)

        if not point_clouds:
            return np.array([]), None

        # Этап 2: Объединение с простым выравниванием
        merged_cloud = self._merge_clouds_fast(point_clouds)

        # Этап 3: Агрессивная очистка
        cleaned_cloud = self._clean_point_cloud(merged_cloud)

        # Этап 4: Создание mesh (опционально)
        # mesh = None
        # if len(cleaned_cloud.points) > 100:
        #    mesh = self._create_mesh(cleaned_cloud, method)

        return np.asarray(cleaned_cloud.points)  # , mesh

    def _create_point_clouds_vectorized(self, selected_masks: List[Dict]) -> List[o3d.geometry.PointCloud]:
        point_clouds = []
        
        # Создаем pointcloud объект RealSense один раз вне цикла
        pc = rs.pointcloud()
        
        for mask_data in selected_masks:
            mask = mask_data['mask']
            intrinsics = mask_data['depth_intrinsics']
            y_coords, x_coords = np.where(mask > 0)
            
            if len(x_coords) < self.min_points_threshold:
                continue
            
            if intrinsics:
                # Используем RealSense деproекцию с интринсиками
                filtered_points = []
                
                for y, x, depth in zip(y_coords, x_coords, depth_values):
                    if depth > 0:
                        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                        point_m = [p / 1000.0 for p in point]
                        filtered_points.append(point_m)
                
                if len(filtered_points) > 0:
                    filtered_points = np.array(filtered_points)
                    valid = (filtered_points[:, 2] > 0) & (filtered_points[:, 2] < 5.0)
                    filtered_points = filtered_points[valid]
                    
                    if len(filtered_points) > 100:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filtered_points)
                        point_clouds.append(pcd)
            else:
                # Fallback на Open3D метод
                depth_image = np.zeros_like(mask, dtype=np.float32)
                depth_image[y_coords, x_coords] = depth_values
                
                depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
                pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_o3d, self.o3d_intrinsics, depth_scale=1000.0, stride=Config.POINT_CLOUD_STRIDE
                )
                
                points = np.asarray(pcd.points)
                valid = (points[:, 2] > 0) & (points[:, 2] < 5.0)
                pcd = pcd.select_by_index(np.where(valid)[0])
                
                if len(pcd.points) > 100:
                    point_clouds.append(pcd)
        
        return point_clouds

    def _merge_clouds_fast(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Быстрое объединение облаков без сложного выравнивания"""
        if len(point_clouds) == 1:
            return point_clouds[0]

        # Простое объединение
        merged = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            merged += pcd

        return merged

    def _clean_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Агрессивная очистка облака точек"""
        if len(pcd.points) == 0:
            return pcd

        # 1. Удаление статистических выбросов
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)

        # 2. Вокселизация для уменьшения шума
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 3. Удаление радиусных выбросов
        pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=0.01)

        return pcd

    def _create_mesh(self, pcd: o3d.geometry.PointCloud, method: str) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Создание mesh различными методами

        Args:
            pcd: Облако точек
            method: Метод реконструкции поверхности
        """
        if len(pcd.points) < 100:
            return None

        # Оценка нормалей (необходимо для большинства методов)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=20)
        )

        # Ориентация нормалей
        pcd.orient_normals_consistent_tangent_plane(k=10)

        try:
            if method == 'poisson':
                # Poisson Surface Reconstruction - самый качественный
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, width=0, scale=1.1, linear_fit=False
                )

                # Удаление низкоплотных вершин
                vertices = np.asarray(mesh.vertices)
                if len(vertices) > 0:
                    bbox = pcd.get_axis_aligned_bounding_box()
                    mesh = mesh.crop(bbox)

            elif method == 'ball_pivoting':
                # Ball Pivoting Algorithm - быстрый и точный
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [0.5 * avg_dist, avg_dist, 2 * avg_dist, 4 * avg_dist]

                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

            elif method == 'alpha_shape':
                # Alpha Shape - хорош для объектов с отверстиями
                # Создаем тетраэдральную сетку
                tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(
                    pcd)

                # Подбираем оптимальный alpha
                alpha = 0.03  # Может потребоваться настройка
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map
                )

            else:  # delaunay
                # 2.5D Delaunay - простой и быстрый
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=7, width=0, scale=1.1, linear_fit=True
                )

            # Постобработка mesh
            if mesh is not None and len(mesh.vertices) > 0:
                mesh = self._postprocess_mesh(mesh)

            return mesh

        except Exception as e:
            print(f"Ошибка создания mesh методом {method}: {e}")
            return None

    def _postprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Постобработка mesh"""
        # Удаление вырожденных треугольников
        mesh.remove_degenerate_triangles()

        # Удаление дублированных треугольников
        mesh.remove_duplicated_triangles()

        # Удаление дублированных вершин
        mesh.remove_duplicated_vertices()

        # Удаление неманифолдных граней
        mesh.remove_non_manifold_edges()

        # Сглаживание (опционально)
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)

        # Вычисление нормалей
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        return mesh

    def get_object_dimensions(self, points: np.ndarray) -> Dict[str, float]:
        """Получение габаритов объекта"""
        if len(points) == 0:
            return {'width': 0, 'height': 0, 'depth': 0, 'volume': 0}

        # Ориентированная ограничивающая рамка
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        obb = pcd.get_oriented_bounding_box()
        extent = obb.extent

        return {
            'width': float(extent[0]),
            'height': float(extent[1]),
            'depth': float(extent[2]),
            'volume': float(extent[0] * extent[1] * extent[2])
        }

    def get_mesh_properties(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        """Получение свойств mesh"""
        if mesh is None or len(mesh.vertices) == 0:
            return {'volume': 0, 'surface_area': 0}

        try:
            # Объем (только для manifold mesh)
            if mesh.is_watertight():
                volume = mesh.get_volume()
            else:
                volume = 0

            # Площадь поверхности
            surface_area = mesh.get_surface_area()

            return {
                'volume': float(volume),
                'surface_area': float(surface_area),
                'vertices_count': len(mesh.vertices),
                'triangles_count': len(mesh.triangles),
                'is_watertight': mesh.is_watertight()
            }
        except:
            return {'volume': 0, 'surface_area': 0}

    def create_mesh(self, points: np.ndarray, method='poisson') -> o3d.geometry.TriangleMesh:
        """Создание меша из массива точек"""
        if len(points) < 100:
            return None

        # Создаем облако точек Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Оценка нормалей
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30)
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
