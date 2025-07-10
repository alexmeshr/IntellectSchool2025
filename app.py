"""Flask приложение"""
from flask import Flask, Response, render_template, jsonify, request
import cv2
import time
from camera_manager import CameraManager
from realsense_camera import RealSenseCamera
from mock_camera import MockCamera
from config import Config
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Глобальный менеджер камеры
camera_manager = None

def generate_frames():
    """Генератор кадров для стриминга"""
    global camera_manager
    while camera_manager and camera_manager.running:
        results = camera_manager.get_processed_frame()
        if results is not None:
            frame = results['combined']
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1.0 / Config.FPS)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global camera_manager
    if camera_manager:
        data = request.json if request.is_json else {}
        record_separate = data.get('separate', False)
        save_raw = data.get('save_raw', True)
        success, message = camera_manager.start_recording(record_separate, save_raw)
        return jsonify({'success': success, 'message': message})
    return jsonify({'success': False, 'message': 'Камера не инициализирована'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global camera_manager
    if camera_manager:
        success, message = camera_manager.stop_recording()
        return jsonify({'success': success, 'message': message})
    return jsonify({'success': False, 'message': 'Камера не инициализирована'})

@app.route('/recording_status')
def recording_status():
    global camera_manager
    if camera_manager:
        return jsonify({
            'recording': camera_manager.recording,
            'separate': camera_manager.record_separate
        })
    return jsonify({'recording': False, 'separate': False})

@app.route('/recordings')
def get_recordings():
    global camera_manager
    if camera_manager:
        files = camera_manager.get_recordings_list()
        return jsonify({'files': files})
    return jsonify({'files': []})

def main(use_mock=False, rgb_folder=None, depth_folder=None):
    """Основная функция запуска"""
    global camera_manager
    
    try:
        # Выбор камеры
        if use_mock:
            print("Использование mock камеры для тестирования")
            camera = MockCamera(rgb_folder, depth_folder)
        else:
            print("Инициализация Intel RealSense...")
            camera = RealSenseCamera()
        
        # Создание менеджера
        camera_manager = CameraManager(camera, Config.CAMERA_INTRINSICS)
        camera_manager.start()
        
        print("Система готова к работе")
        print("Открывайте браузер: http://localhost:5000")
        
        # Запуск Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
    finally:
        if camera_manager:
            camera_manager.stop()

@app.route('/processed_objects')
def get_processed_objects():
    global camera_manager
    if camera_manager and camera_manager.frame_processor:
        # Получаем ограничения габаритов из параметров запроса
        max_length = float(request.args.get('max_length', 55)) / 100  # переводим в метры
        max_width = float(request.args.get('max_width', 40)) / 100
        max_height = float(request.args.get('max_height', 20)) / 100
        max_dims = [max_length, max_width, max_height]
        
        # Получаем завершенные объекты
        completed_objects = camera_manager.frame_processor.get_completed_objects()
        
        # Подготавливаем данные для отправки
        objects_data = []
        for obj in completed_objects:
            # Создаем изображение с визуализацией, передавая ограничения
            annotated_image = camera_manager.frame_processor.create_object_visualization(
                obj, max_dims
            )
            
            # Конвертируем в base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            objects_data.append({
                'id': obj['id'],
                'dimensions': obj['dimensions'].tolist(),  # [длина, ширина, высота] в метрах
                'volume': float(obj['volume']),
                'timestamp': obj['timestamp'].isoformat(),
                'image_url': f'data:image/jpeg;base64,{img_base64}'
            })
        
        # Очищаем список после отправки
        camera_manager.frame_processor.completed_objects.clear()
        
        return jsonify({'objects': objects_data})
    
    return jsonify({'objects': []})

# Добавить endpoint для очистки объектов
@app.route('/clear_objects', methods=['POST'])
def clear_objects():
    global camera_manager
    if camera_manager and camera_manager.frame_processor:
        camera_manager.frame_processor.completed_objects.clear()
        return jsonify({'success': True})
    return jsonify({'success': False})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RealSense Person Removal System')
    parser.add_argument('--mock', action='store_true', help='Использовать mock камеру')
    parser.add_argument('--video', type=str, help='Путь к видео для mock камеры')
    parser.add_argument('--depth', type=str, help='Путь к depth видео для mock камеры')
    
    args = parser.parse_args()
    main(use_mock=args.mock, rgb_folder=args.video, depth_folder=args.depth)

