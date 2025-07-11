"""Оптимизированная реализация интерфейса камеры для Intel RealSense"""
import pyrealsense2 as rs
import numpy as np
from interfaces import ICameraInterface
from config import Config
import cv2
import threading
import queue
import time
from collections import deque

class RealSenseCamera(ICameraInterface):
    def __init__(self, apply_filters=True, buffer_size=100):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.error_count = 0
        self.max_errors = 5
        
        # Буферизация кадров
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Потоковая обработка
        self.capture_thread = None
        self.is_capturing = False
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Статистика производительности
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Настройка потоков с оптимизацией
        self.config.enable_stream(rs.stream.depth,
                                  Config.DEPTH_RESOLUTION[0],
                                  Config.DEPTH_RESOLUTION[1],
                                  rs.format.z16, Config.FPS)
        self.config.enable_stream(rs.stream.color,
                                  Config.RGB_RESOLUTION[0],
                                  Config.RGB_RESOLUTION[1],
                                  rs.format.bgr8, Config.FPS)
        self.apply_filters = apply_filters

    def setup_filters(self):
        """Настройка оптимизированных фильтров"""
        if self.apply_filters:
            # Уменьшаем нагрузку фильтров для повышения производительности
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 1)  # Уменьшено с 2
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.3)  # Уменьшено
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 15)  # Уменьшено

            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.2)  # Уменьшено
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 15)  # Уменьшено

            self.hole_filling_filter = rs.hole_filling_filter()
            # Используем более быстрый режим заполнения дырок
            self.hole_filling_filter.set_option(rs.option.holes_fill, 1)  # Farest from around

            self.threshold_filter = rs.threshold_filter()
            self.threshold_filter.set_option(rs.option.min_distance, 0.15)
            self.threshold_filter.set_option(rs.option.max_distance, 4.0)

    def start(self):
        """Запуск камеры с оптимизированными настройками"""
        try:
            # Настройка контекста для повышения производительности
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("RealSense устройство не найдено")

            profile = self.pipeline.start(self.config)
            device = profile.get_device()
            
            # Оптимизация настроек устройства
            self._optimize_device_settings(device)
            
            # Настройка фильтров
            self.setup_filters()
            
            # Запуск потока захвата кадров
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Получение интринсиков
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print(f"Камера запущена. Intrinsics: fx={self.depth_intrinsics.fx:.1f}, "
                  f"fy={self.depth_intrinsics.fy:.1f}, ppx={self.depth_intrinsics.ppx:.1f}, "
                  f"ppy={self.depth_intrinsics.ppy:.1f}")
            
            # Прогрев камеры
            self._warmup_camera()
            
        except Exception as e:
            print(f"Ошибка запуска камеры: {e}")
            raise

    def _optimize_device_settings(self, device):
        """Оптимизация настроек устройства для минимизации лагов"""
        try:
            # Настройки глубины
            depth_sensor = device.first_depth_sensor()
            
            if depth_sensor.supports(rs.option.visual_preset):
                # Используем balanced preset для компромисса между качеством и скоростью
                depth_sensor.set_option(rs.option.visual_preset, 
                                      rs.rs400_visual_preset.default)
            
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, Config.LASER_POWER)
            
            if depth_sensor.supports(rs.option.confidence_threshold):
                depth_sensor.set_option(rs.option.confidence_threshold, Config.LASER_TRESHOLD)
            
            # Отключаем автоэкспозицию для стабильности FPS
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                
            # Настройка экспозиции вручную для стабильности
            if depth_sensor.supports(rs.option.exposure):
                depth_sensor.set_option(rs.option.exposure, 8500)  # Фиксированная экспозиция
            
            # Настройки RGB камеры
            color_sensor = device.first_color_sensor()
            if color_sensor:
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                    
        except Exception as e:
            print(f"Предупреждение: не удалось оптимизировать настройки устройства: {e}")

    def _warmup_camera(self):
        """Прогрев камеры для стабилизации"""
        print("Прогрев камеры...")
        warmup_frames = 30
        for i in range(warmup_frames):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if i % 10 == 0:
                    print(f"Прогрев: {i}/{warmup_frames}")
            except:
                pass
        print("Прогрев завершен")

    def _capture_frames(self):
        """Поток захвата кадров"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while self.is_capturing:
            try:
                # Захват кадров с таймаутом
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                
                if not frames:
                    continue
                
                # Сброс счетчика ошибок при успешном захвате
                consecutive_errors = 0
                
                # Выравнивание кадров
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Применение фильтров
                if self.apply_filters:
                    depth_frame = self._apply_filters(depth_frame)
                
                # Преобразование в numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Поворот если нужно
                if not Config.ORIENTATION_VERTICAL:
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
                    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
                
                # Создание пакета кадров
                frame_package = {
                    'color': color_image,
                    'depth': depth_image,
                    'intrinsics': self.depth_intrinsics,
                    'timestamp': time.time()
                }
                
                # Добавление в буфер (неблокирующее)
                try:
                    self.frame_queue.put_nowait(frame_package)
                    self.frame_count += 1
                except queue.Full:
                    # Удаляем старый кадр если буфер полный
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_package)
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                
                # Обновление статистики FPS
                self._update_fps_stats()
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Ошибка захвата кадра: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Слишком много последовательных ошибок, попытка перезапуска...")
                    self._restart_camera()
                    consecutive_errors = 0
                
                time.sleep(0.01)  # Небольшая задержка при ошибке

    def _apply_filters(self, depth_frame):
        """Применение фильтров к кадру глубины"""
        try:
            # Применяем фильтры последовательно
            filtered_frame = self.threshold_filter.process(depth_frame)
            filtered_frame = self.spatial_filter.process(filtered_frame)
            filtered_frame = self.temporal_filter.process(filtered_frame)
            filtered_frame = self.hole_filling_filter.process(filtered_frame)
            return filtered_frame
        except Exception as e:
            print(f"Ошибка применения фильтров: {e}")
            return depth_frame

    def _update_fps_stats(self):
        """Обновление статистики FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            if self.frame_count > 0:
                drop_rate = (self.dropped_frames / self.frame_count) * 100
                print(f"FPS: {self.current_fps:.1f}, Dropped: {drop_rate:.1f}%")
            
            self.frame_count = 0
            self.dropped_frames = 0
            self.last_fps_time = current_time

    def get_frames(self):
        """Получить последние RGB и depth кадры из буфера"""
        try:
            # Получение кадра из очереди (неблокирующее)
            frame_package = self.frame_queue.get_nowait()
            return (frame_package['color'], 
                   frame_package['depth'], 
                   frame_package['intrinsics'])
        except queue.Empty:
            # Возвращаем None если нет доступных кадров
            return None, None, None

    def get_latest_frame(self):
        """Получить самый последний кадр, пропуская промежуточные"""
        latest_frame = None
        try:
            # Извлекаем все кадры до самого последнего
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame:
                return (latest_frame['color'], 
                       latest_frame['depth'], 
                       latest_frame['intrinsics'])
        except:
            pass
        
        return None, None, None

    def get_frame_with_timeout(self, timeout=0.1):
        """Получить кадр с таймаутом"""
        try:
            frame_package = self.frame_queue.get(timeout=timeout)
            return (frame_package['color'], 
                   frame_package['depth'], 
                   frame_package['intrinsics'])
        except queue.Empty:
            return None, None, None

    def is_frame_available(self):
        """Проверить доступность кадра"""
        return not self.frame_queue.empty()

    def get_buffer_size(self):
        """Получить текущий размер буфера"""
        return self.frame_queue.qsize()

    def stop(self):
        print("Остановка камеры...")
        self.is_capturing = False
        
        # Добавить эти строки:
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)  # Увеличить таймаут
            if self.capture_thread.is_alive():
                print("Предупреждение: поток захвата не завершился")
        
        
        # Ожидание завершения потока захвата
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Остановка пайплайна
        try:
            self.pipeline.stop()
        except:
            pass
        
        # Очистка буфера
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        print("Камера остановлена")

    def _restart_camera(self):
        """Перезапуск камеры при критических ошибках"""
        try:
            print("Перезапуск камеры...")
            
            # Остановка текущего захвата
            was_capturing = self.is_capturing
            self.is_capturing = False
            
            # Остановка пайплайна
            try:
                self.pipeline.stop()
            except:
                pass
            
            # Задержка для освобождения ресурсов
            time.sleep(1.0)
            
            # Переинициализация
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
            
            # Запуск если камера была активна
            if was_capturing:
                profile = self.pipeline.start(self.config)
                device = profile.get_device()
                self._optimize_device_settings(device)
                self.setup_filters()
                
                # Получение интринсиков
                depth_stream = profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                
                # Запуск потока захвата
                self.is_capturing = True
                self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
                self.capture_thread.start()
            
            self.error_count = 0
            print("Камера успешно перезапущена")
            
        except Exception as e:
            print(f"Критическая ошибка при перезапуске: {e}")
            raise

    def get_stats(self):
        """Получить статистику работы камеры"""
        return {
            'fps': self.current_fps,
            'buffer_size': self.get_buffer_size(),
            'is_capturing': self.is_capturing,
            'error_count': self.error_count
        }
    
    def __del__(self):
        """Деструктор для гарантированной очистки ресурсов"""
        try:
            self.stop()
        except:
            pass