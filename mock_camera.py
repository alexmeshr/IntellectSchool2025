"""Mock камера для тестирования без реального оборудования"""
import numpy as np
import cv2
from interfaces import ICameraInterface
from config import Config

class MockCamera(ICameraInterface):
    """Эмулятор камеры для тестирования"""
    def __init__(self, video_path=None, depth_path=None):
        self.video_path = video_path
        self.depth_path = depth_path
        self.cap_video = None
        self.cap_depth = None
        
    def start(self):
        if self.video_path:
            self.cap_video = cv2.VideoCapture(self.video_path)
        if self.depth_path:
            self.cap_depth = cv2.VideoCapture(self.depth_path)
    
    def get_frames(self):
        # Генерация тестовых кадров или чтение из файлов
        if self.cap_video and self.cap_video.isOpened():
            ret, color_image = self.cap_video.read()
            if not ret:
                self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, color_image = self.cap_video.read()
        else:
            # Генерация тестового изображения
            color_image = np.random.randint(0, 255, 
                (*Config.RGB_RESOLUTION[::-1], 3), dtype=np.uint8)
        
        if self.cap_depth and self.cap_depth.isOpened():
            ret, depth_frame = self.cap_depth.read()
            if not ret:
                self.cap_depth.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, depth_frame = self.cap_depth.read()
            depth_image = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 100
        else:
            # Генерация тестовой карты глубины
            depth_image = np.random.randint(0, 5000, 
                Config.DEPTH_RESOLUTION[::-1], dtype=np.uint16)
        
        return color_image, depth_image
    
    def stop(self):
        if self.cap_video:
            self.cap_video.release()
        if self.cap_depth:
            self.cap_depth.release()

