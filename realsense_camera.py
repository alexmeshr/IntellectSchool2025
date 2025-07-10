"""Реализация интерфейса камеры для Intel RealSense"""
import pyrealsense2 as rs
import numpy as np
from interfaces import ICameraInterface
from config import Config
import cv2


class RealSenseCamera(ICameraInterface):
    def __init__(self, apply_filters=True):
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
        self.apply_filters = apply_filters

    def setup_filters(self):
        """Настройка фильтров для улучшения качества глубины"""
        # Пространственный фильтр
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

        # Временной фильтр
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        # Фильтр дырок
        self.hole_filling_filter = rs.hole_filling_filter()

        # Децимация (уменьшение разрешения для ускорения)
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 1)

        # Пороговый фильтр
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0.15)
        self.threshold_filter.set_option(rs.option.max_distance, 4.0)

    def start(self):
        """Запуск камеры с оптимальными настройками"""
        profile = self.pipeline.start(self.config)

        # Настройки датчика глубины
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.visual_preset):
            # rs.rs400_visual_preset.default - настройки по умолчанию
            # rs.rs400_visual_preset.high_accuracy - высокая точность
            # rs.rs400_visual_preset.high_density - высокая плотность точек
            # rs.rs400_visual_preset.medium_density - средняя плотность
            depth_sensor.set_option(rs.option.visual_preset,
                                    rs.rs400_visual_preset.high_accuracy)

        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, Config.LASER_POWER)

        if depth_sensor.supports(rs.option.confidence_threshold):
            # Порог уверенности (1-3, где 3 - самый строгий)
            depth_sensor.set_option(
                rs.option.confidence_threshold, Config.LASER_TRESHOLD)

        # Включение пост-обработки
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.setup_filters()

        intr = profile.get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # rs.set(intrinsics.Width,480)
        # rs.set(intrinsics.Height,640)
        print("ppx is: ", intr.ppx)
        print("ppy is: ", intr.ppy)
        print("fx is: ", intr.fx)
        print("fy is: ", intr.fy)
        # print("cx is: ",intr.cx)
        # print("cy is: ",intr.cy)

    def get_frames(self):
        """Получить выровненные RGB и depth кадры"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            if self.apply_filters:
                depth_frame = self.decimation_filter.process(depth_frame)
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling_filter.process(depth_frame)
                depth_frame = self.threshold_filter.process(depth_frame)

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
