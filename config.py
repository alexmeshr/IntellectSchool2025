"""Конфигурация приложения"""
import os

class Config:
    # Разрешения камеры
    RGB_RESOLUTION = (640, 480)
    DEPTH_RESOLUTION = (640, 480)
    FPS = 15
    
    # Настройки глубины
    DEPTH_SCALE = 0.03
    LASER_POWER = 240
    
    # Настройки YOLO
    YOLO_MODEL = 'yolov8l-seg.pt'
    YOLO_CONFIDENCE = 0.5
    BAGGAGE_CONFIDENCE = 0.2
    
    # Классы для детекции
    PERSON_CLASS = 0
    BAGGAGE_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
    
    # Пути
    OUTPUT_DIR = "recordings"
    RAW_DATA_DIR = "raw_data"
    
    # Настройки записи
    VIDEO_CODEC = 'mp4v'
    JPEG_QUALITY = 70
    
    @staticmethod
    def ensure_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)

