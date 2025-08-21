#!/usr/bin/env python3
"""
Примеры использования программы колоризации видео
Демонстрирует различные сценарии и возможности
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from colorization_model import VideoColorizer
from utils import get_video_info, is_grayscale_video, create_comparison_image
import cv2
import numpy as np


def example_1_basic_colorization():
    """Пример 1: Базовая колоризация одного кадра"""
    print("🎨 Пример 1: Базовая колоризация кадра")
    print("-" * 40)
    
    # Создаем простое черно-белое изображение
    height, width = 300, 400
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Добавляем различные объекты
    cv2.circle(test_image, (100, 100), 50, (128, 128, 128), -1)  # Серый круг
    cv2.rectangle(test_image, (200, 150), (350, 250), (200, 200, 200), -1)  # Светло-серый прямоугольник
    cv2.line(test_image, (0, 200), (width, 200), (64, 64, 64), 5)  # Темно-серая линия
    
    # Сохраняем тестовое изображение
    cv2.imwrite("test_bw_image.jpg", test_image)
    print("✅ Создано тестовое черно-белое изображение: test_bw_image.jpg")
    
    # Инициализируем колоризатор
    try:
        colorizer = VideoColorizer(device='cpu')  # Используем CPU для примера
        
        # Колоризуем кадр
        colorized = colorizer.colorize_frame(test_image)
        
        # Сохраняем результат
        cv2.imwrite("test_colorized_image.jpg", colorized)
        print("✅ Колоризованное изображение сохранено: test_colorized_image.jpg")
        
        # Создаем сравнение
        comparison = create_comparison_image(test_image, colorized, "comparison_example1.jpg")
        print("✅ Сравнение сохранено: comparison_example1.jpg")
        
    except Exception as e:
        print(f"❌ Ошибка в примере 1: {e}")


def example_2_video_analysis():
    """Пример 2: Анализ видео файла"""
    print("\n🔍 Пример 2: Анализ видео файла")
    print("-" * 40)
    
    # Создаем тестовое видео если его нет
    test_video = "example_video.mp4"
    
    if not os.path.exists(test_video):
        print("📹 Создаем тестовое видео...")
        create_simple_test_video(test_video)
    
    try:
        # Получаем информацию о видео
        info = get_video_info(test_video)
        
        print(f"📊 Информация о видео {test_video}:")
        print(f"   Разрешение: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']:.2f}")
        print(f"   Кадров: {info['frame_count']}")
        print(f"   Длительность: {info['duration']:.2f} сек")
        print(f"   Размер файла: {info['file_size']:.2f} MB")
        
        # Проверяем, черно-белое ли видео
        is_bw = is_grayscale_video(test_video, sample_frames=5)
        print(f"   Черно-белое: {'Да' if is_bw else 'Нет'}")
        
    except Exception as e:
        print(f"❌ Ошибка в примере 2: {e}")


def example_3_batch_processing():
    """Пример 3: Пакетная обработка нескольких кадров"""
    print("\n📦 Пример 3: Пакетная обработка кадров")
    print("-" * 40)
    
    try:
        # Создаем несколько тестовых кадров
        frames = []
        for i in range(3):
            frame = create_test_frame(i)
            frames.append(frame)
            cv2.imwrite(f"test_frame_{i}.jpg", frame)
        
        print(f"✅ Создано {len(frames)} тестовых кадров")
        
        # Инициализируем колоризатор
        colorizer = VideoColorizer(device='cpu')
        
        # Обрабатываем каждый кадр
        for i, frame in enumerate(frames):
            print(f"🎨 Обрабатываем кадр {i+1}/{len(frames)}...")
            
            colorized = colorizer.colorize_frame(frame)
            output_path = f"colorized_frame_{i}.jpg"
            cv2.imwrite(output_path, colorized)
            
            print(f"   ✅ Сохранен: {output_path}")
        
        print("✅ Пакетная обработка завершена")
        
    except Exception as e:
        print(f"❌ Ошибка в примере 3: {e}")


def example_4_custom_settings():
    """Пример 4: Использование пользовательских настроек"""
    print("\n⚙️ Пример 4: Пользовательские настройки")
    print("-" * 40)
    
    try:
        # Различные настройки устройства
        devices = ['cpu']
        
        # Добавляем GPU если доступен
        try:
            import torch
            if torch.cuda.is_available():
                devices.append('cuda')
        except ImportError:
            pass
        
        for device in devices:
            print(f"🖥️  Тестируем устройство: {device}")
            
            # Создаем колоризатор с указанным устройством
            colorizer = VideoColorizer(device=device)
            
            # Создаем тестовый кадр
            test_frame = create_test_frame(0)
            
            # Засекаем время
            import time
            start_time = time.time()
            
            # Колоризуем
            colorized = colorizer.colorize_frame(test_frame)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"   ⏱️  Время обработки: {processing_time:.3f} сек")
            
            # Сохраняем результат
            output_path = f"colorized_{device}.jpg"
            cv2.imwrite(output_path, colorized)
            print(f"   ✅ Результат сохранен: {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка в примере 4: {e}")


def create_test_frame(frame_id):
    """Создает тестовый кадр с различными объектами"""
    height, width = 240, 320
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Фон с градиентом
    for y in range(height):
        intensity = int(255 * (y / height))
        frame[y, :] = intensity
    
    # Объекты зависят от ID кадра
    if frame_id == 0:
        # Круги разного размера
        cv2.circle(frame, (80, 80), 30, (200, 200, 200), -1)
        cv2.circle(frame, (240, 160), 40, (150, 150, 150), -1)
    elif frame_id == 1:
        # Прямоугольники
        cv2.rectangle(frame, (50, 50), (150, 100), (180, 180, 180), -1)
        cv2.rectangle(frame, (200, 120), (280, 180), (120, 120, 120), -1)
    else:
        # Линии и многоугольники
        points = np.array([[100, 50], [150, 100], [100, 150], [50, 100]], np.int32)
        cv2.fillPoly(frame, [points], (160, 160, 160))
        cv2.line(frame, (0, height//2), (width, height//2), (100, 100, 100), 3)
    
    return frame


def create_simple_test_video(output_path, duration=2, fps=10):
    """Создает простое тестовое видео"""
    width, height = 320, 240
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    for frame_num in range(total_frames):
        # Создаем черно-белый кадр
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Движущийся объект
        t = frame_num / total_frames
        x = int(width * t)
        y = height // 2
        
        cv2.circle(frame, (x, y), 20, 200, -1)
        cv2.rectangle(frame, (10, 10), (50, 50), 150, -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Тестовое видео создано: {output_path}")


def cleanup_examples():
    """Удаляет файлы, созданные в примерах"""
    files_to_remove = [
        "test_bw_image.jpg", "test_colorized_image.jpg", "comparison_example1.jpg",
        "example_video.mp4", "colorized_cpu.jpg", "colorized_cuda.jpg"
    ]
    
    # Добавляем файлы с номерами
    for i in range(3):
        files_to_remove.extend([
            f"test_frame_{i}.jpg",
            f"colorized_frame_{i}.jpg"
        ])
    
    print("\n🧹 Очистка файлов примеров...")
    removed_count = 0
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️  Удален: {file}")
            removed_count += 1
    
    print(f"✅ Удалено файлов: {removed_count}")


def main():
    """Запускает все примеры"""
    print("🎨 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ПРОГРАММЫ КОЛОРИЗАЦИИ")
    print("=" * 60)
    
    try:
        # Запускаем примеры
        example_1_basic_colorization()
        example_2_video_analysis()
        example_3_batch_processing()
        example_4_custom_settings()
        
        print("\n🎉 Все примеры выполнены успешно!")
        
        # Предлагаем очистку
        response = input("\nУдалить созданные файлы? (y/N): ").lower()
        if response == 'y':
            cleanup_examples()
        else:
            print("📁 Файлы примеров сохранены для просмотра")
        
    except KeyboardInterrupt:
        print("\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Примеры использования колоризации видео")
    parser.add_argument("--example", type=int, choices=[1,2,3,4], 
                       help="Запустить конкретный пример")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Удалить файлы примеров")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_examples()
    elif args.example:
        examples = {
            1: example_1_basic_colorization,
            2: example_2_video_analysis,
            3: example_3_batch_processing,
            4: example_4_custom_settings
        }
        examples[args.example]()
    else:
        main()