# frame_processor.py
"""Основной процессор кадров с поддержкой трекинга"""
import cv2
import numpy as np
from datetime import datetime
from baggage_detector import BaggageDetector
from baggage_tracker import BaggageTracker
from point_cloud_reconstructor import PointCloudReconstructor
from depth_processor import DepthProcessor
from config import Config
from dimension_estimator import DimensionEstimator
import copy
import threading


def non_max_suppression_bbox(detections, iou_threshold=0.5):
    boxes = [d['bbox'] for d in detections]
    scores = [d.get('confidence', 1.0) for d in detections]

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    indices = indices.flatten() if len(indices) > 0 else []
    return [detections[i] for i in indices]


class FrameProcessor:
    def __init__(self, camera_intrinsics=None):
        self.detector = BaggageDetector()
        self.depth_processor = DepthProcessor()
        self.tracker = BaggageTracker(
            max_disappeared=Config.MAX_DISAPPEARED,
            max_distance=Config.MAX_DISTANCE
        )

        # Инициализируем реконструктор только если переданы параметры камеры
        self.dimension_estimator = None
        if camera_intrinsics is not None:
            self.dimension_estimator = DimensionEstimator(camera_intrinsics)

        self.frame_count = 0
        self.enable_tracking = True  # Флаг для включения/выключения трекинга
        self.completed_objects = []
        self.completed_objects_lock = threading.Lock()

    def add_completed_object(self, obj_data):
        """Потокобезопасное добавление объекта"""
        with self.completed_objects_lock:
            self.completed_objects.append(obj_data)
            print(f"[COMPLETED] Добавлен объект {obj_data['id']} в список завершенных")
            
    def get_completed_objects(self):
        """Потокобезопасное получение и очистка списка объектов"""
        with self.completed_objects_lock:
            objects = self.completed_objects.copy()
            self.completed_objects.clear()
            return objects

    def process_frame(self, color_image, depth_image, depth_intrinsics, is_calibration=False):
        """Обработка кадра: детекция, сегментация, очистка depth"""
        self.frame_count += 1
        timestamp = datetime.now()

        # Детекция и сегментация
        detection_results = self.detector.detect_and_segment(color_image, is_calibration)
        person_mask = detection_results['person_mask']
        baggage_mask = detection_results['baggage_mask']

        # Очистка карты глубины - оставляем только багаж
        cleaned_depth = self.depth_processor.keep_only_baggage(
            depth_image, baggage_mask)

        # Создание визуализаций
        depth_colormap = self.depth_processor.create_colormap(depth_image)
        cleaned_depth_colormap = self.depth_processor.create_colormap(
            cleaned_depth)

        # Трекинг багажа (если включен)
        tracked_objects = {}
        if self.enable_tracking:
            # Сначала проверяем объекты на финальную обработку
            finalized_objects = self.tracker.get_and_clear_finalized_objects()
            # Обрабатываем объекты перед обновлением трекера
            for obj in finalized_objects:
                self._finalize_object(obj)

            # Обновляем трекер
            filtered_detections = non_max_suppression_bbox(
                detection_results['baggage_detections'])
            tracked_objects = self.tracker.update(filtered_detections)

            # Сохраняем depth маски для активных объектов
            for obj_id, tracked_obj in tracked_objects.items():
                if tracked_obj.mask is not None:
                    # Извлекаем значения глубины под маской
                    depth_values = depth_image[tracked_obj.mask > 0]
                    rgb_values = color_image  # [tracked_obj.mask > 0]
                    if len(depth_values) > 0:
                        tracked_obj.add_depth_observation(
                            tracked_obj.mask,
                            depth_values,
                            copy.deepcopy(rgb_values),
                            timestamp,
                            depth_intrinsics
                        )

        # Визуализация масок на RGB
        if self.enable_tracking and tracked_objects:
            # Визуализация с трекингом
            color_with_mask = self._visualize_with_tracking(
                color_image, person_mask, tracked_objects
            )
        else:
            # Оригинальная визуализация без трекинга
            color_with_mask = self._visualize_detections(
                color_image, person_mask, baggage_mask
            )

        # Объединение изображений
        combined = np.hstack((color_with_mask, cleaned_depth_colormap))

        return {
            'color_with_mask': color_with_mask,
            'depth_colormap': depth_colormap,
            'cleaned_depth_colormap': cleaned_depth_colormap,
            'combined': combined,
            'person_mask': person_mask,
            'baggage_mask': baggage_mask,
            'cleaned_depth': cleaned_depth,
            'tracked_objects': tracked_objects,
            'frame_count': self.frame_count
        }

    def _visualize_detections(self, color_image, person_mask, baggage_mask):
        """Оригинальная визуализация детекций на изображении"""
        color_with_mask = color_image.copy()

        # Полупрозрачная зеленая маска для людей, синяя для багажа
        overlay = color_image.copy()
        overlay[person_mask > 0] = [0, 255, 0]
        overlay[baggage_mask > 0] = [0, 0, 255]
        color_with_mask = cv2.addWeighted(color_image, 0.5, overlay, 0.3, 0)

        # Контуры людей (зеленые)
        contours, _ = cv2.findContours(
            person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_with_mask, contours, -1, (0, 255, 0), 2)

        # Контуры багажа (синие)
        contours, _ = cv2.findContours(
            baggage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_with_mask, contours, -1, (0, 0, 255), 2)

        return color_with_mask

    def _visualize_with_tracking(self, color_image, person_mask, tracked_objects):
        """Визуализация с трекингом багажа"""
        color_with_mask = color_image.copy()

        # Полупрозрачная зеленая маска для людей
        overlay = color_image.copy()
        overlay[person_mask > 0] = [0, 255, 0]

        # Применяем маску людей
        color_with_mask = cv2.addWeighted(color_image, 0.7, overlay, 0.3, 0)

        # Контуры людей (зеленые)
        person_contours, _ = cv2.findContours(
            person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(color_with_mask, person_contours, -1, (0, 255, 0), 2)

        # Отслеживаемый багаж с уникальными цветами для каждого ID
        for obj_id, tracked_obj in tracked_objects.items():
            # Получаем уникальный цвет для ID
            color = self._get_color_for_id(obj_id)

            # Полупрозрачная маска для багажа
            overlay = color_with_mask.copy()
            overlay[tracked_obj.mask > 0] = color
            color_with_mask = cv2.addWeighted(
                color_with_mask, 0.7, overlay, 0.3, 0)

            # Контур маски
            contours, _ = cv2.findContours(
                tracked_obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(color_with_mask, contours, -1, color, 2)

            # Центроид
            # if tracked_obj.centroid:
            #    cv2.circle(color_with_mask, tracked_obj.centroid, 4, color, -1)
            #    cv2.circle(color_with_mask, tracked_obj.centroid, 5, (255, 255, 255), 1)

        return color_with_mask

    def _get_color_for_id(self, obj_id):
        """Генерация уникального цвета для ID"""
        # Используем HSV для ярких различимых цветов
        hue = (obj_id * 45) % 180  # Распределяем по цветовому кругу
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)

    def set_tracking_enabled(self, enabled):
        """Включение/выключение трекинга"""
        self.enable_tracking = enabled
        if not enabled:
            # Очищаем трекер при выключении
            self.tracker = BaggageTracker(
                max_disappeared=Config.MAX_DISAPPEARED,
                max_distance=Config.MAX_DISTANCE
            )

    def _finalize_object(self, tracked_obj):
        """Финальная обработка объекта перед удалением"""
        if len(tracked_obj.depth_masks) < Config.MIN_OBSERVATIONS:
            print(
                f"Object {tracked_obj.id}: недостаточно масок ({len(tracked_obj.depth_masks)})")
            return

        # Используем DimensionEstimator для оценки габаритов
        if self.dimension_estimator:
            result = self.dimension_estimator.estimate_dimensions(
                tracked_obj.depth_masks)

            if result is None:
                print(f"Object {tracked_obj.id}: не удалось оценить габариты")
                return

            # Сохраняем результат
            self.completed_objects.append({
                'id': tracked_obj.id,
                'unique_id': tracked_obj.unique_id,
                'all_masks': tracked_obj.depth_masks,
                # [длина, ширина, высота] в метрах
                'dimensions': result['dimensions'],
                'volume': result['volume'],
                'point_cloud': result['point_cloud'],
                'centroid': result['centroid'],
                'rotation_matrix': result['rotation_matrix'],
                'best_frame_index': result['frame_index'],
                'num_points': result['num_points'],
                'rgb_image': result['rgb_image'],
                'timestamp': datetime.now()
            })

            print(f"Object {tracked_obj.id}: габариты {result['dimensions'][0]:.3f}x{result['dimensions'][1]:.3f}x{result['dimensions'][2]:.3f} м, "
                  f"объем {result['volume']:.4f} м³, {result['num_points']} точек")
        else:
            print(
                f"Object {tracked_obj.id}: DimensionEstimator не инициализирован")

    # Добавить в класс FrameProcessor новый метод
    def create_object_visualization(self, obj_data, max_dims_meters):
        """Создание изображения с визуализацией объекта и его габаритов

        Args:
            obj_data: данные объекта
            max_dims_meters: максимальные габариты в метрах [длина, ширина, высота]
        """
        # Берем RGB изображение из лучшего кадра
        rgb_image = obj_data['rgb_image'].copy()

        # Получаем маску из лучшего кадра
        best_frame_index = obj_data['best_frame_index']
        mask_data = obj_data['all_masks'][best_frame_index]
        mask = mask_data['mask']

        # Находим контур объекта
        contours, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)

            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Габариты в см для отображения
            dimensions_cm = [dim * 100 for dim in obj_data['dimensions']]

            # Проверяем, помещается ли в допустимые габариты
            fits = self._check_dimensions_fit(
                obj_data['dimensions'], max_dims_meters)

            # Цвет в зависимости от результата проверки
            color = (0, 255, 0) if fits else (0, 0, 255)  # Зеленый или красный

            # Рисуем контур
            cv2.drawContours(rgb_image, [largest_contour], -1, color, 2)

            # Подготавливаем текст с габаритами
            text = f"{dimensions_cm[0]:.1f}x{dimensions_cm[1]:.1f}x{dimensions_cm[2]:.1f} cm"

            # Находим позицию для текста
            text_y = y - 10 if y > 30 else y + h + 25

            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(rgb_image,
                          (x, text_y - text_height - 5),
                          (x + text_width + 10, text_y + 5),
                          (255, 255, 255), -1)

            # Текст с габаритами
            cv2.putText(rgb_image, text, (x + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return rgb_image

    def _check_dimensions_fit(self, dims, max_dims):
        """Проверка, помещаются ли габариты в допустимые размеры"""
        # Сортируем оба набора размеров по убыванию
        sorted_dims = sorted(dims, reverse=True)
        sorted_max = sorted(max_dims, reverse=True)

        # Проверяем, что каждый размер помещается
        for i in range(3):
            if sorted_dims[i] > sorted_max[i]:
                return False
        return True

    @property
    def camera_intrinsics(self):
        """Получение параметров камеры"""
        if self.reconstructor:
            return self.reconstructor.intrinsics
        return None

