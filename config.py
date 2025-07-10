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
    MAX_DISTANCE = 100  # Максимальное расстояние для сопоставления (пиксели)
    
    # Параметры 3D реконструкции
    MIN_OBSERVATIONS = 5  # Минимум наблюдений для реконструкции
    MAX_OBSERVATIONS = 15  # Максимум для оптимизации памяти
    POINT_CLOUD_STRIDE = 1
    TOP_N_ESTIMATIONS = 6
    OUTLIER_THRESHOLD = 2
    
    # Пути
    OUTPUT_DIR = "recordings"
    RAW_DATA_DIR = "raw_data"
    POINT_CLOUDS_DIR = "point_clouds"
    
    # Настройки записи
    VIDEO_CODEC = 'mp4v'
    JPEG_QUALITY = 70
    ORIENTATION_VERTICAL = True 

    CAMERA_INTRINSICS = {
        'fx': 390.443,      # фокусное расстояние по X
        'fy': 390.443,      # фокусное расстояние по Y
        'cx': 319.957,      # главная точка X (ppx в RealSense)
        'cy': 243.082,      # главная точка Y (ppy в RealSense)
        'depth_scale': 0.001 # масштаб глубины (метры)
    }
    LASER_TRESHOLD = 2
    
    @staticmethod
    def ensure_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(Config.POINT_CLOUDS_DIR, exist_ok=True)

