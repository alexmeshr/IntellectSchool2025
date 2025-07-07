"""Реализация интерфейса камеры для Intel RealSense"""
import pyrealsense2 as rs
import numpy as np
from interfaces import ICameraInterface
from config import Config
import cv2

class RealSenseCamera(ICameraInterface):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        
        # Настройка потоков
        self.config.enable_stream(rs.stream.depth, 
                                Config.DEPTH_RESOLUTION[0], 
                                Config.DEPTH_RESOLUTION[1], 
                                rs.format.z16, Config.FPS)
        self.config.enable_stream(rs.stream.color, 
                                Config.RGB_RESOLUTION[0], 
                                Config.RGB_RESOLUTION[1], 
                                rs.format.bgr8, Config.FPS)
        
    def start(self):
        """Запуск камеры с оптимальными настройками"""
        profile = self.pipeline.start(self.config)
        
        # Настройки датчика глубины
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, True)
        
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, Config.LASER_POWER)
    
    def get_frames(self):
        """Получить выровненные RGB и depth кадры"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if Config.ROTATE:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
                color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Ошибка получения кадров: {e}")
            return None, None
    
    def stop(self):
        """Остановка камеры"""
        self.pipeline.stop()
