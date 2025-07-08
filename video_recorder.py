"""Модуль для записи видео"""
import cv2
import numpy as np
import os
from datetime import datetime
from config import Config

class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.writers = {}
        self.raw_writers = {}
        self.filenames = {}
        self.frame_count = 0
        Config.ensure_directories()
        
    def start_recording(self, record_separate=False, save_raw=True):
        """Начать запись видео"""
        if self.recording:
            return False, "Запись уже идет"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.frame_count = 0
            
            # Подготовка имен файлов
            self.filenames = {
                'combined': f"{Config.OUTPUT_DIR}/combined_{timestamp}.mp4",
                'rgb': f"{Config.OUTPUT_DIR}/rgb_{timestamp}.mp4",
                'depth': f"{Config.OUTPUT_DIR}/depth_{timestamp}.mp4",
                'cleaned': f"{Config.OUTPUT_DIR}/cleaned_{timestamp}.mp4",
                'raw_rgb': f"{Config.RAW_DATA_DIR}/raw_rgb_{timestamp}.mp4",
                'raw_depth': f"{Config.RAW_DATA_DIR}/raw_depth_{timestamp}.avi"
            }
            
            self.record_separate = record_separate
            self.save_raw = save_raw
            self.recording = True
            
            return True, f"Запись начата: {timestamp}"
            
        except Exception as e:
            return False, f"Ошибка начала записи: {e}"
    
    def write_frame(self, color_image, depth_image, color_with_mask, 
                   depth_colormap, cleaned_depth_colormap, combined):
        """Записать кадр во все необходимые потоки"""
        if not self.recording:
            return
        
        try:
            self.frame_count += 1
            
            # Инициализация писателей при первом кадре
            if not self.writers:
                h, w = color_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
                
                if self.record_separate:
                    self.writers['rgb'] = cv2.VideoWriter(
                        self.filenames['rgb'], fourcc, Config.FPS, (w, h))
                    self.writers['depth'] = cv2.VideoWriter(
                        self.filenames['depth'], fourcc, Config.FPS, (w, h))
                    self.writers['cleaned'] = cv2.VideoWriter(
                        self.filenames['cleaned'], fourcc, Config.FPS, (w, h))
                else:
                    h_combined, w_combined = combined.shape[:2]
                    self.writers['combined'] = cv2.VideoWriter(
                        self.filenames['combined'], fourcc, Config.FPS, (w_combined, h_combined))
                
                # Писатели для сырых данных
                if self.save_raw:
                    self.raw_writers['rgb'] = cv2.VideoWriter(
                        self.filenames['raw_rgb'], fourcc, Config.FPS, (w, h))
                    # Для depth используем lossless кодек
                    self.raw_writers['depth'] = cv2.VideoWriter(
                        self.filenames['raw_depth'], 
                        cv2.VideoWriter_fourcc(*'FFV1'), Config.FPS, (w, h), False)
            
            # Запись обработанных кадров
            if self.record_separate:
                self.writers['rgb'].write(color_with_mask)
                self.writers['depth'].write(depth_colormap)
                self.writers['cleaned'].write(cleaned_depth_colormap)
            else:
                self.writers['combined'].write(combined)
            
            # Запись сырых данных
            if self.save_raw:
                self.raw_writers['rgb'].write(color_image)
                # Нормализация depth для записи в 8-bit
                depth_normalized = (depth_image / depth_image.max() * 255).astype(np.uint8)
                self.raw_writers['depth'].write(depth_normalized)
                
                # Дополнительно сохраняем каждый N-й кадр как .npy для точных данных
                if self.frame_count % 30 == 0:  # Каждые 2 секунды при 15 FPS
                    np.save(f"{Config.RAW_DATA_DIR}/depth_frame_{self.frame_count:06d}.npy", 
                           depth_image)
                
        except Exception as e:
            print(f"Ошибка записи кадра: {e}")
    
    def stop_recording(self):
        """Остановить запись"""
        if not self.recording:
            return False, "Запись не идет"
        
        try:
            self.recording = False
            
            # Закрытие всех писателей
            for writer in self.writers.values():
                if writer:
                    writer.release()
            
            for writer in self.raw_writers.values():
                if writer:
                    writer.release()
            
            self.writers = {}
            self.raw_writers = {}
            
            files = [os.path.basename(f) for f in self.filenames.values() 
                    if os.path.exists(f)]
            
            return True, f"Записано: {', '.join(files)}"
            
        except Exception as e:
            return False, f"Ошибка остановки записи: {e}"

        def process_and_write_from_files(self, rgb_folder, depth_npy_folder):
            """Обрабатывает и записывает RGB и глубинные кадры из файлов"""
            rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg'))])
            depth_files = sorted([f for f in os.listdir(depth_npy_folder) if f.endswith('.npy')])

            if len(rgb_files) != len(depth_files):
                print("Количество RGB и depth кадров не совпадает!")
                return

            for rgb_file, depth_file in zip(rgb_files, depth_files):
                try:
                    # Загрузка изображений
                    color_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
                    depth_image = np.load(os.path.join(depth_npy_folder, depth_file))

                    # Глубина в colormap
                    #depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    #depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                    # Комбинированное изображение (эмуляция)
                    combined = np.hstack((color_image.copy(), color_image.copy(), color_image.copy()))

                    # Запись
                    self.write_frame(color_image, depth_image, color_with_mask,
                                    depth_colormap, cleaned_depth_colormap, combined)

                except Exception as e:
                    print(f"Ошибка обработки кадра {rgb_file}: {e}")

    
    @staticmethod
    def get_recordings_list():
        """Получить список записанных файлов"""
        try:
            files = []
            for directory in [Config.OUTPUT_DIR, Config.RAW_DATA_DIR]:
                if not os.path.exists(directory):
                    continue
                    
                for filename in os.listdir(directory):
                    if filename.endswith(('.mp4', '.avi', '.npy')):
                        filepath = os.path.join(directory, filename)
                        size = os.path.getsize(filepath)
                        mtime = os.path.getmtime(filepath)
                        files.append({
                            'name': filename,
                            'path': directory,
                            'size': f"{size / (1024*1024):.1f} MB",
                            'date': datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                        })
            return sorted(files, key=lambda x: x['date'], reverse=True)
        except Exception as e:
            print(f"Ошибка получения списка файлов: {e}")
            return []
