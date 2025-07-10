"""Mock камера для тестирования без реального оборудования"""
import numpy as np
import cv2
import os
from interfaces import ICameraInterface
from config import Config


class MockCamera(ICameraInterface):
    """Эмулятор камеры для тестирования"""

    def __init__(self, rgb_folder=None, depth_folder=None):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.rgb_files = []
        self.depth_files = []
        self.current_frame = 0

    def start(self):
        if self.rgb_folder and os.path.exists(self.rgb_folder):
            self.rgb_files = sorted([f for f in os.listdir(self.rgb_folder)
                                     if f.endswith('.png')])

        if self.depth_folder and os.path.exists(self.depth_folder):
            self.depth_files = sorted([f for f in os.listdir(self.depth_folder)
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

        # Переход к следующему кадру (с циклом)
        self.current_frame += 1
        if self.rgb_files and self.current_frame >= len(self.rgb_files):
            self.current_frame = 0

        return color_image, depth_image

    def stop(self):
        self.rgb_files = []
        self.depth_files = []
        self.current_frame = 0
