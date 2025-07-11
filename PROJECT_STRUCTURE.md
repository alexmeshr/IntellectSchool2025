# Структура проекта

## 📁 Файловая структура

```
baggage-dimension-system/
├── app.py                    # Flask веб-приложение и API endpoints
├── config.py                 # Конфигурация системы (разрешения, параметры детекции)
├── interfaces.py             # Абстрактные интерфейсы для камеры и детектора
├── camera_manager.py         # Главный менеджер системы, координирует все компоненты
├── realsense_camera.py       # Драйвер для Intel RealSense D435
├── mock_camera.py            # Эмулятор камеры для тестирования без оборудования
├── baggage_detector.py       # YOLO детекция и сегментация людей/багажа
├── baggage_tracker.py        # Трекинг объектов между кадрами (Kalman filter)
├── depth_processor.py        # Обработка карт глубины, удаление людей
├── frame_processor.py        # Основная обработка кадров, координация pipeline
├── dimension_estimator.py    # Оценка габаритов по 3D данным
├── point_cloud_reconstructor.py  # 3D реконструкция (если есть)
├── video_recorder.py         # Запись обработанных и сырых данных
├── templates/
│   └── index.html           # Веб-интерфейс с видеопотоком и управлением
├── static/                  # Статические файлы (CSS, JS, изображения)
├── recordings/              # Директория для записанных обработанных видео
├── raw_data/               # Сырые данные (RGB, depth, intrinsics)
├── point_clouds/           # Экспортированные 3D модели багажа
└── README.md               # Основная документация проекта
```

## 📦 Полный список зависимостей

### Основные библиотеки

```bash
pip install pyrealsense2        # Intel RealSense SDK для работы с камерой
pip install numpy              # Численные вычисления и работа с массивами
pip install opencv-python      # Компьютерное зрение и обработка изображений
pip install flask             # Веб-фреймворк для интерфейса
pip install ultralytics       # YOLOv8 для детекции и сегментации
pip install open3d            # Работа с облаками точек и 3D
pip install scipy             # Научные вычисления и статистика
```

### Дополнительные зависимости (могут потребоваться)

```bash
pip install filterpy          # Фильтры Калмана для трекинга (если используется)
pip install Pillow            # Работа с изображениями
pip install matplotlib        # Визуализация данных (для отладки)
```

### Установка одной командой

```bash
pip install pyrealsense2 numpy opencv-python flask ultralytics open3d scipy filterpy Pillow matplotlib
```

## 🔧 Описание основных модулей

### Ядро системы

- **`camera_manager.py`** - Центральный координатор, управляет потоком данных между всеми компонентами
- **`frame_processor.py`** - Обрабатывает каждый кадр: детекция → трекинг → измерение габаритов
- **`config.py`** - Все настройки в одном месте (разрешения, пороги детекции, параметры трекинга)

### Работа с камерой

- **`realsense_camera.py`** - Реализация для Intel RealSense с фильтрами улучшения глубины
- **`mock_camera.py`** - Позволяет работать без камеры, используя записанные данные
- **`interfaces.py`** - Абстракции для легкой замены источника данных

### Детекция и трекинг

- **`baggage_detector.py`** - YOLOv8 находит людей и багаж, создает маски сегментации
- **`baggage_tracker.py`** - Отслеживает объекты между кадрами, использует Калман для предсказания движения
- **`depth_processor.py`** - Удаляет людей из карты глубины, оставляя только багаж

### Измерение габаритов

- **`dimension_estimator.py`** - Выбирает лучший кадр, строит 3D модель, вычисляет oriented bounding box
- **`point_cloud_reconstructor.py`** - Дополнительные методы 3D реконструкции (опционально)

### Интерфейс и запись

- **`app.py`** - Flask сервер с API endpoints и веб-интерфейсом
- **`video_recorder.py`** - Сохраняет обработанные видео и сырые данные для анализа
- **`templates/index.html`** - Веб-интерфейс с live preview и управлением

## 💾 Форматы данных

### Входные данные
- **RGB**: 640×480, 15 FPS, BGR формат
- **Depth**: 640×480, 16-bit, миллиметры

### Сохраняемые данные
- **Обработанные видео**: MP4 с визуализацией масок и габаритов
- **Сырые RGB**: папка с кадрами
- **Сырые Depth**: папка с npy массивами глубин для каждого кадра

## 🚀 Быстрый старт для разработчиков

### Минимальный пример использования

```python
from camera_manager import CameraManager
from realsense_camera import RealSenseCamera
from config import Config

# Инициализация
camera = RealSenseCamera()
manager = CameraManager(camera, Config.CAMERA_INTRINSICS)

# Запуск
manager.start()

# Получение обработанного кадра
results = manager.get_processed_frame()
if results:
    cv2.imshow('Result', results['combined'])
    
# Остановка
manager.stop()
```

### Работа с mock данными

```python
from mock_camera import MockCamera

# Использование записанных данных
mock_camera = MockCamera(
    rgb_folder='./raw_data/rgb_frames/',
    depth_folder='./raw_data/depth_frames/',
    intrinsics_folder='./raw_data/intrinsics/'
)
```

## 🐛 Отладка

### Полезные флаги в config.py

```python
ORIENTATION_VERTICAL = True  # Поворот изображения на 90°
MIN_OBSERVATIONS = 5        # Минимум кадров для измерения
OUTLIER_THRESHOLD = 2       # Порог для фильтрации аномальных измерений
```
