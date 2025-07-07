"""Конфигурация приложения"""
import os

class Config:
    # Разрешения камеры
    RGB_RESOLUTION = (640, 480)
    DEPTH_RESOLUTION = (640, 480)
    FPS = 15
    ROTATE=True
    
    # Настройки глубины
    DEPTH_SCALE = 0.03
    LASER_POWER = 240
    
    # Настройки YOLO
    YOLO_MODEL = "yolov8l-seg.pt" # #
    BAG_CONFIDENCE = 0.2
    PERSON_CONFIDENCE = 0.99 #отключил людей
    
    # Классы для детекции
    PERSON_CLASS = 0
    BAGGAGE_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase

    # Параметры трекинга
    MAX_DISAPPEARED = 10  # Кадров до удаления объекта
    MAX_DISTANCE = 50  # Максимальное расстояние для сопоставления (пиксели)
    
    # Параметры 3D реконструкции
    MIN_OBSERVATIONS = 5  # Минимум наблюдений для реконструкции
    MAX_OBSERVATIONS = 30  # Максимум для оптимизации памяти

    
    # Пути
    OUTPUT_DIR = "recordings"
    RAW_DATA_DIR = "raw_data"
    POINT_CLOUDS_DIR = "point_clouds"
    
    # Настройки записи
    VIDEO_CODEC = 'mp4v'
    JPEG_QUALITY = 70
    
    @staticmethod
    def ensure_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(Config.POINT_CLOUDS_DIR, exist_ok=True)

