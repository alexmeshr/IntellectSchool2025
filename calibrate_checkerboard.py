#!/usr/bin/env python3
"""Скрипт калибровки RealSense по шахматной доске"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from datetime import datetime

PATTERN_SIZE = (9, 6)  # Размер доски (внутренние углы)
SQUARE_SIZE = 0.027    # Размер клетки в метрах


def main():
    # Настройка pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Запуск камеры
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Ввод параметров
    param = input(
        "Введите количество фото N или путь к папке с изображениями: ").strip()

    if param.isdigit():
        # Режим съемки
        n_photos = int(param)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"calibration_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nБудет сделано {n_photos} фото. Нажимайте Enter для съемки.")
        print("Держите шахматную доску под разными углами!")
        print("ESC - выход, Enter - съемка\n")

        images = []
        PATTERN_SIZE = (9, 6)

        for i in range(n_photos):
            print(f"Готовится фото {i+1}/{n_photos}...")

            while True:
                # Получение кадров
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                display = color_image.copy()

                # Поиск доски для подсказки
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, PATTERN_SIZE, None)

                if ret:
                    cv2.drawChessboardCorners(
                        display, PATTERN_SIZE, corners, ret)
                    cv2.putText(display, "Board detected! Press Enter", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No board detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(display, f"Photo {i+1}/{n_photos}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('RealSense Calibration', display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    sys.exit(0)
                elif key == 13:  # Enter
                    break

            # Сохранение кадра
            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())

            cv2.imwrite(f"{output_dir}/color_{i:03d}.jpg", color_image)
            np.save(f"{output_dir}/depth_{i:03d}.npy", depth_image)
            images.append(color_image)

            # Подтверждение
            cv2.putText(display, "SAVED!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('RealSense Calibration', display)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    else:
        # Режим загрузки из папки
        output_dir = param
        if not os.path.exists(output_dir):
            print(f"Папка {output_dir} не найдена!")
            return

        images = []
        for f in sorted(os.listdir(output_dir)):
            if f.startswith('color_') and f.endswith('.jpg'):
                img = cv2.imread(os.path.join(output_dir, f))
                images.append(img)

        print(f"Загружено {len(images)} изображений")

    # Калибровка
    print("\nВыполняется калибровка...")
    # Критерии для уточнения углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Подготовка точек
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                           0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"✓ Изображение {i+1}: доска найдена")
        else:
            print(f"✗ Изображение {i+1}: доска не найдена")

    if len(objpoints) < 3:
        print("\nНедостаточно изображений с доской для калибровки!")
        return

    # Калибровка
    print(f"\nКалибровка по {len(objpoints)} изображениям...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Оценка ошибки
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    avg_error = total_error / len(objpoints)

    # Результаты
    print("\n=== РЕЗУЛЬТАТЫ КАЛИБРОВКИ ===")
    print(f"fx: {mtx[0, 0]:.2f}")
    print(f"fy: {mtx[1, 1]:.2f}")
    print(f"cx: {mtx[0, 2]:.2f}")
    print(f"cy: {mtx[1, 2]:.2f}")
    print(f"Искажения: {dist.flatten()}")
    print(f"Средняя ошибка: {avg_error:.3f} пикселей")

    # Сохранение
    calib_file = os.path.join(output_dir, "calibration.npz")
    np.savez(calib_file,
             fx=mtx[0, 0],
             fy=mtx[1, 1],
             cx=mtx[0, 2],
             cy=mtx[1, 2],
             dist_coeffs=dist,
             matrix=mtx,
             error=avg_error)

    print(f"\nКалибровка сохранена в {calib_file}")

    # Завершение
    pipeline.stop()


if __name__ == "__main__":
    main()
