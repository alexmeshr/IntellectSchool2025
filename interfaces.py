"""Интерфейсы для абстракции зависимостей"""
from abc import ABC, abstractmethod
import numpy as np


class ICameraInterface(ABC):
    @abstractmethod
    def get_frames(self):
        """Получить RGB и depth кадры"""
        pass

    @abstractmethod
    def start(self):
        """Запустить камеру"""
        pass

    @abstractmethod
    def stop(self):
        """Остановить камеру"""
        pass


class IPersonDetector(ABC):
    @abstractmethod
    def detect_and_segment(self, image: np.ndarray) -> tuple:
        """Детектировать и сегментировать людей
        Returns: (person_mask, baggage_boxes)
        """
        pass
