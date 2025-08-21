"""
Утилиты для работы с изображениями и видео
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import color
import torch


def is_grayscale_video(video_path, sample_frames=10):
    """
    Проверяет, является ли видео черно-белым
    
    Args:
        video_path: путь к видео файлу
        sample_frames: количество кадров для анализа
    
    Returns:
        bool: True если видео черно-белое
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // sample_frames)
    
    grayscale_count = 0
    frames_checked = 0
    
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Проверяем, одинаковы ли все каналы цвета
        b, g, r = cv2.split(frame)
        
        # Если каналы практически одинаковые, считаем кадр черно-белым
        if np.allclose(b, g, atol=5) and np.allclose(g, r, atol=5):
            grayscale_count += 1
        
        frames_checked += 1
        
        if frames_checked >= sample_frames:
            break
    
    cap.release()
    
    # Считаем видео черно-белым, если больше 80% кадров черно-белые
    return (grayscale_count / frames_checked) > 0.8


def create_comparison_image(original_frame, colorized_frame, output_path=None):
    """
    Создает изображение для сравнения оригинального и колоризованного кадров
    
    Args:
        original_frame: оригинальный кадр
        colorized_frame: колоризованный кадр
        output_path: путь для сохранения (опционально)
    
    Returns:
        numpy.ndarray: изображение сравнения
    """
    # Убеждаемся, что размеры совпадают
    h, w = original_frame.shape[:2]
    colorized_resized = cv2.resize(colorized_frame, (w, h))
    
    # Создаем изображение сравнения
    comparison = np.hstack([original_frame, colorized_resized])
    
    # Добавляем текст
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Colorized', (w + 10, 30), font, 1, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, comparison)
    
    return comparison


def extract_sample_frames(video_path, output_dir, num_frames=5):
    """
    Извлекает образцы кадров из видео для предварительного просмотра
    
    Args:
        video_path: путь к видео
        output_dir: директория для сохранения кадров
        num_frames: количество кадров для извлечения
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // (num_frames + 1)  # +1 чтобы избежать последнего кадра
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    
    for i in range(1, num_frames + 1):
        frame_pos = i * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if ret:
            frame_path = output_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(str(frame_path))
            print(f"Извлечен кадр {i}: {frame_path}")
    
    cap.release()
    return extracted_frames


def get_video_info(video_path):
    """
    Получает информацию о видео файле
    
    Args:
        video_path: путь к видео
    
    Returns:
        dict: словарь с информацией о видео
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'file_size': os.path.getsize(video_path) / (1024 * 1024)  # MB
    }
    
    cap.release()
    return info


def create_preview_grid(frames, output_path, grid_size=(2, 3)):
    """
    Создает сетку из кадров для предварительного просмотра
    
    Args:
        frames: список кадров (numpy arrays)
        output_path: путь для сохранения
        grid_size: размер сетки (rows, cols)
    """
    rows, cols = grid_size
    
    if len(frames) > rows * cols:
        frames = frames[:rows * cols]
    
    # Изменяем размер всех кадров до одинакового
    target_height, target_width = 200, 300
    resized_frames = []
    
    for frame in frames:
        resized = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized)
    
    # Дополняем черными кадрами если нужно
    while len(resized_frames) < rows * cols:
        black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        resized_frames.append(black_frame)
    
    # Создаем сетку
    grid_rows = []
    for i in range(rows):
        row_frames = resized_frames[i * cols:(i + 1) * cols]
        row = np.hstack(row_frames)
        grid_rows.append(row)
    
    grid = np.vstack(grid_rows)
    
    # Сохраняем
    cv2.imwrite(output_path, grid)
    print(f"Сетка предварительного просмотра сохранена: {output_path}")


def check_system_resources():
    """
    Проверяет системные ресурсы для оптимальной работы
    
    Returns:
        dict: информация о системе
    """
    import psutil
    
    # Проверяем GPU
    gpu_available = torch.cuda.is_available()
    gpu_memory = 0
    gpu_name = "Не обнаружен"
    
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_name = torch.cuda.get_device_name(0)
    
    # Проверяем RAM
    ram = psutil.virtual_memory()
    
    # Проверяем CPU
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    info = {
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_memory_gb': gpu_memory,
        'ram_total_gb': ram.total / (1024**3),
        'ram_available_gb': ram.available / (1024**3),
        'cpu_cores': cpu_count,
        'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0
    }
    
    return info


def print_system_info():
    """Выводит информацию о системе"""
    try:
        info = check_system_resources()
        
        print("🖥️  ИНФОРМАЦИЯ О СИСТЕМЕ")
        print("=" * 40)
        print(f"GPU: {info['gpu_name']}")
        if info['gpu_available']:
            print(f"GPU память: {info['gpu_memory_gb']:.1f} GB")
        print(f"RAM: {info['ram_available_gb']:.1f} GB / {info['ram_total_gb']:.1f} GB")
        print(f"CPU: {info['cpu_cores']} ядер @ {info['cpu_freq_mhz']:.0f} MHz")
        print("=" * 40)
        
        # Рекомендации
        if info['gpu_available'] and info['gpu_memory_gb'] >= 4:
            print("✅ Система оптимальна для колоризации видео")
        elif info['ram_available_gb'] >= 8:
            print("⚠️  GPU недоступен, будет использован CPU (медленнее)")
        else:
            print("⚠️  Недостаточно ресурсов для оптимальной работы")
        
    except ImportError:
        print("⚠️  Модуль psutil недоступен для проверки системы")


if __name__ == "__main__":
    # Тестирование утилит
    print("🔧 Тестирование утилит...")
    print_system_info()