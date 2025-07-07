# Система определения габаритов багажа для летней школы «Интеллект» 2025

## Установка

```bash
# Установка зависимостей
pip install pyrealsense2 numpy opencv-python flask ultralytics
```

## Структура проекта

```
├── config.py           # Конфигурация системы
├── interfaces.py       # Интерфейсы для абстракции
├── person_detector.py  # Модуль детекции (YOLO)
├── depth_processor.py  # Обработка карт глубины
├── realsense_camera.py # Драйвер Intel RealSense
├── mock_camera.py      # Mock камера для тестов
├── frame_processor.py  # Обработчик кадров
├── video_recorder.py   # Запись видео
├── camera_manager.py   # Менеджер системы
├── app.py             # Flask приложение
└── templates/
    └── index.html     # Веб-интерфейс
```

## Использование

### Запуск с реальной камерой

```bash
python app.py
```

### Запуск с mock камерой для тестирования

```bash
# С использованием записанных видео
python app.py --mock --video path/to/rgb.mp4 --depth path/to/depth.avi
# Например
python app.py --mock --video .\\raw_data\\raw_rgb_20250707_082118.mp4 --depth .\\raw_data\\raw_depth_20250707_082118.avi
```

### Использование отдельных модулей для тестирования

```python
from person_detector import YOLOPersonDetector
from depth_processor import DepthProcessor
import cv2

# Детекция людей на изображении
detector = YOLOPersonDetector()
image = cv2.imread('test.jpg')
mask, boxes = detector.detect_and_segment(image)

# Обработка карты глубины
processor = DepthProcessor()
cleaned = processor.remove_person_from_depth(depth_image, mask)
```

### Работа с записанными данными

Система сохраняет:
- **recordings/** - обработанные видео (с масками, очищенные depth)
- **raw_data/** - исходные данные для последующего анализа
  - raw_rgb_*.mp4 - исходное RGB видео
  - raw_depth_*.avi - видео карт глубины (FFV1 кодек)
  - depth_frame_*.npy - точные NumPy массивы depth (каждые 30 кадров)

### Пример обработки записанных данных

```python
import numpy as np
import cv2
from frame_processor import FrameProcessor

# Загрузка точных данных depth
depth_data = np.load('raw_data/depth_frame_000030.npy')

# Загрузка видео
cap = cv2.VideoCapture('raw_data/raw_rgb_20240101_120000.mp4')
ret, frame = cap.read()

# Обработка
processor = FrameProcessor()
results = processor.process_frame(frame, depth_data)
```

## API endpoints

- `GET /` - веб-интерфейс
- `GET /video_feed` - видеопоток
- `POST /start_recording` - начать запись
  - `separate`: bool - записывать отдельные файлы
  - `save_raw`: bool - сохранять исходные данные
- `POST /stop_recording` - остановить запись
- `GET /recording_status` - статус записи
- `GET /recordings` - список записанных файлов

## Конфигурация

Основные параметры в `config.py`:

```python
RGB_RESOLUTION = (640, 480)
DEPTH_RESOLUTION = (640, 480)
FPS = 15
YOLO_MODEL = 'yolov8l-seg.pt'
PERSON_CLASS = 0
BAGGAGE_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
```

## Требования

- Python 3.7+
- Intel RealSense SDK 2.0
- CUDA-совместимый GPU (опционально, для ускорения YOLO)
