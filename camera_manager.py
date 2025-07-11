"""Менеджер камеры - основной класс для работы с системой"""
import threading
import time
from interfaces import ICameraInterface
from frame_processor import FrameProcessor
from video_recorder import VideoRecorder
from config import Config


class CameraManager:
    def __init__(self, camera: ICameraInterface, camera_intrinsics=None):
        self.camera = camera

        self._frame_processor = FrameProcessor(camera_intrinsics)
        self.video_recorder = VideoRecorder()
        self.running = False

    def export_tracked_objects(self):
        """Экспорт облаков точек отслеживаемых объектов"""
        return self._frame_processor.export_point_clouds()

    def start(self):
        """Запуск системы"""
        Config.ensure_directories()
        self.camera.start()
        self.running = True

    def stop(self):
        """Остановка системы"""
        if self.video_recorder.recording:
            self.video_recorder.stop_recording()
        self.running = False
        self.camera.stop()

    def get_processed_frame(self):
        """Получить обработанный кадр"""
        if not self.running:
            return None

        # Получение кадров с камеры
        color_image, depth_image, depth_intrinsics = self.camera.get_frames()
        if color_image is None or depth_image is None:
            return None

        # Обработка кадров
        results = self._frame_processor.process_frame(color_image, depth_image, depth_intrinsics)

        # Запись если включена
        self.video_recorder.write_frame(
            color_image, depth_image, depth_intrinsics,
            results['color_with_mask'],
            results['depth_colormap'],
            results['cleaned_depth_colormap'],
            results['combined']
        )

        return results

    def start_recording(self, record_separate=False, save_raw=True):
        """Начать запись"""
        return self.video_recorder.start_recording(record_separate, save_raw)

    def stop_recording(self):
        """Остановить запись"""
        return self.video_recorder.stop_recording()

    @property
    def recording(self):
        return self.video_recorder.recording

    @property
    def record_separate(self):
        return getattr(self.video_recorder, 'record_separate', False)

    @property
    def frame_processor(self):
        return self._frame_processor

    def get_recordings_list(self):
        return VideoRecorder.get_recordings_list()
