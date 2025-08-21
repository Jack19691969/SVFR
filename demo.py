#!/usr/bin/env python3
"""
Демонстрационный скрипт для программы колоризации видео
Создает тестовое черно-белое видео и демонстрирует колоризацию
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def create_test_video(output_path="test_bw_video.mp4", duration=5, fps=30):
    """
    Создает тестовое черно-белое видео с различными сценами
    
    Args:
        output_path: путь для сохранения видео
        duration: длительность в секундах
        fps: кадры в секунду
    """
    width, height = 640, 480
    total_frames = duration * fps
    
    # Настройка кодека
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    print(f"Создаем тестовое видео: {output_path}")
    print(f"Параметры: {width}x{height}, {fps} FPS, {duration} сек")
    
    for frame_num in range(total_frames):
        # Создаем кадр с различными паттернами
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Фон с градиентом
        for y in range(height):
            frame[y, :] = int(255 * (y / height))
        
        # Движущиеся объекты
        t = frame_num / total_frames
        
        # Движущийся круг
        center_x = int(width * (0.2 + 0.6 * t))
        center_y = height // 3
        cv2.circle(frame, (center_x, center_y), 30, 200, -1)
        
        # Движущийся прямоугольник
        rect_x = int(width * (0.8 - 0.6 * t))
        rect_y = 2 * height // 3
        cv2.rectangle(frame, (rect_x - 40, rect_y - 20), (rect_x + 40, rect_y + 20), 150, -1)
        
        # Статические элементы
        cv2.line(frame, (0, height // 2), (width, height // 2), 100, 2)
        
        # Текст с номером кадра
        text = f"Frame {frame_num + 1}/{total_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # Добавляем шум для реалистичности
        noise = np.random.randint(-20, 20, (height, width), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
        
        # Показываем прогресс
        if (frame_num + 1) % (fps // 2) == 0:
            progress = (frame_num + 1) / total_frames * 100
            print(f"Прогресс: {progress:.1f}%")
    
    out.release()
    print(f"✅ Тестовое видео создано: {output_path}")
    return output_path


def run_demo():
    """Запускает полную демонстрацию"""
    print("🎬 ДЕМОНСТРАЦИЯ ПРОГРАММЫ КОЛОРИЗАЦИИ ВИДЕО")
    print("=" * 50)
    
    # Создаем тестовое видео
    test_video = create_test_video("demo_input.mp4", duration=3, fps=15)
    
    # Проверяем, что основные файлы существуют
    required_files = ["main.py", "colorization_model.py", "utils.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        return
    
    print("\n🔧 Проверяем системные ресурсы...")
    try:
        from utils import print_system_info
        print_system_info()
    except ImportError as e:
        print(f"⚠️  Не удалось проверить систему: {e}")
    
    print(f"\n🎨 Запускаем колоризацию тестового видео...")
    print(f"Команда: python main.py {test_video} -o demo_output.mp4 --preview")
    
    # Запускаем колоризацию
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "main.py", test_video, 
            "-o", "demo_output.mp4", "--preview", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Демонстрация завершена успешно!")
            print("\nФайлы:")
            print(f"  📁 Входное видео: {test_video}")
            print(f"  📁 Выходное видео: demo_output.mp4")
            
            # Показываем информацию о файлах
            if os.path.exists("demo_output.mp4"):
                input_size = os.path.getsize(test_video) / 1024  # KB
                output_size = os.path.getsize("demo_output.mp4") / 1024  # KB
                print(f"\nРазмеры файлов:")
                print(f"  📊 Входной: {input_size:.1f} KB")
                print(f"  📊 Выходной: {output_size:.1f} KB")
        else:
            print("❌ Ошибка при колоризации:")
            print(result.stderr)
            
    except FileNotFoundError:
        print("❌ Python не найден или main.py недоступен")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")


def create_comparison_demo():
    """Создает демонстрационное сравнение до/после"""
    print("\n📊 Создаем демонстрационное сравнение...")
    
    if not os.path.exists("demo_input.mp4") or not os.path.exists("demo_output.mp4"):
        print("⚠️  Демонстрационные видео не найдены. Запустите сначала основную демонстрацию.")
        return
    
    # Извлекаем по одному кадру из каждого видео
    cap_input = cv2.VideoCapture("demo_input.mp4")
    cap_output = cv2.VideoCapture("demo_output.mp4")
    
    # Берем кадр из середины
    frame_pos = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
    
    cap_input.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    cap_output.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    
    ret1, frame1 = cap_input.read()
    ret2, frame2 = cap_output.read()
    
    if ret1 and ret2:
        # Конвертируем черно-белый кадр в RGB для сравнения
        if len(frame1.shape) == 2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        
        # Создаем сравнение
        comparison = np.hstack([frame1, frame2])
        
        # Добавляем подписи
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original (B&W)', (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, 'Colorized', (frame1.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite("demo_comparison.jpg", comparison)
        print("✅ Сравнение сохранено: demo_comparison.jpg")
    
    cap_input.release()
    cap_output.release()


def cleanup_demo_files():
    """Удаляет демонстрационные файлы"""
    demo_files = ["demo_input.mp4", "demo_output.mp4", "demo_comparison.jpg"]
    
    print("\n🧹 Очистка демонстрационных файлов...")
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  🗑️  Удален: {file}")


def main():
    parser = argparse.ArgumentParser(description="Демонстрация программы колоризации видео")
    parser.add_argument("--cleanup", action="store_true", help="Удалить демонстрационные файлы")
    parser.add_argument("--create-video", action="store_true", help="Только создать тестовое видео")
    parser.add_argument("--comparison", action="store_true", help="Только создать сравнение")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_demo_files()
        return
    
    if args.create_video:
        create_test_video()
        return
    
    if args.comparison:
        create_comparison_demo()
        return
    
    # Полная демонстрация
    run_demo()
    create_comparison_demo()
    
    print("\n🎉 Демонстрация завершена!")
    print("\nДля очистки демонстрационных файлов запустите:")
    print("python demo.py --cleanup")


if __name__ == "__main__":
    main()