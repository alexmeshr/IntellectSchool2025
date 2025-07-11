"""Mock камера для тестирования без реального оборудования"""
import numpy as np
import cv2
import os
from interfaces import ICameraInterface
from config import Config
import pyrealsense2 as rs

class MockCamera(ICameraInterface):
    """Эмулятор камеры для тестирования"""

    def __init__(self, rgb_folder=None, depth_folder=None, intrinsics_folder=None):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.intrinsics_folder = intrinsics_folder
        self.rgb_files = []
        self.depth_files = []
        self.intrinsics_files = []
        self.current_frame = 0

    def start(self):
        if self.rgb_folder and os.path.exists(self.rgb_folder):
            self.rgb_files = sorted([f for f in os.listdir(self.rgb_folder)
                                     if f.endswith('.png')])

        if self.depth_folder and os.path.exists(self.depth_folder):
            self.depth_files = sorted([f for f in os.listdir(self.depth_folder)
                                       if f.endswith('.npy')])
        if self.intrinsics_folder and os.path.exists(self.intrinsics_folder):
            self.intrinsics_files = sorted([f for f in os.listdir(self.intrinsics_folder)
                                            if f.endswith('.npy')])
        self.current_frame = 0

    def get_frames(self):
        # Чтение RGB кадра
        if self.rgb_files and self.current_frame < len(self.rgb_files):
            rgb_path = os.path.join(
                self.rgb_folder, self.rgb_files[self.current_frame])
            color_image = cv2.imread(rgb_path)
        else:
            # Генерация тестового изображения
            color_image = np.random.randint(0, 255,
                                            (*Config.RGB_RESOLUTION[::-1], 3), dtype=np.uint8)

        # Чтение Depth кадра
        if self.depth_files and self.current_frame < len(self.depth_files):
            depth_path = os.path.join(
                self.depth_folder, self.depth_files[self.current_frame])
            depth_image = np.load(depth_path)
        else:
            # Генерация тестовой карты глубины
            depth_image = np.random.randint(0, 5000,
                                            Config.DEPTH_RESOLUTION[::-1], dtype=np.uint16)

        # чтениe intrinsics
        if self.intrinsics_files and self.current_frame < len(self.intrinsics_files):
            intrinsics_path = os.path.join(
                self.intrinsics_folder, self.intrinsics_files[self.current_frame])
            depth_intrinsics = np.load(intrinsics_path, allow_pickle=True)
            intrinsics_data = np.load(intrinsics_path, allow_pickle=True).item()
            depth_intrinsics = rs.intrinsics()
            depth_intrinsics.width = intrinsics_data['width']
            depth_intrinsics.height = intrinsics_data['height']
            depth_intrinsics.ppx = intrinsics_data['ppx']
            depth_intrinsics.ppy = intrinsics_data['ppy']
            depth_intrinsics.fx = intrinsics_data['fx']
            depth_intrinsics.fy = intrinsics_data['fy']
            #depth_intrinsics.model = rs.distortion.none  # или другая модель
            #depth_intrinsics.coeffs = intrinsics_data.get('coeffs', [0.0] * 5)
        else:
            # Генерация тестовых intrinsics (3x3 матрица)
            depth_intrinsics = np.array([[500, 0, 320],
                                        [0, 500, 240],
                                        [0, 0, 1]], dtype=np.float32)

        # Переход к следующему кадру (с циклом)
        self.current_frame += 1
        if self.rgb_files and self.current_frame >= len(self.rgb_files):
            self.current_frame = 0

        return color_image, depth_image, depth_intrinsics  

    def stop(self):
        self.rgb_files = []
        self.depth_files = []
        self.intrinsics_files = []  
        self.current_frame = 0
