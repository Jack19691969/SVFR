#!/usr/bin/env python3
"""
Программа для колоризации черно-белого видео
Использует глубокое обучение для автоматической колоризации

Автор: AI Assistant
"""

import argparse
import os
import sys
from pathlib import Path
import time

try:
    from colorization_model import VideoColorizer
    from tqdm import tqdm
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что установлены все зависимости: pip install -r requirements.txt")
    sys.exit(1)


def progress_callback(current, total):
    """Колбэк для отображения прогресса"""
    percentage = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rПрогресс: |{bar}| {percentage:.1f}% ({current}/{total})', end='', flush=True)


def validate_input_file(file_path):
    """Проверка входного файла"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Проверяем расширение файла
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        raise ValueError(f"Неподдерживаемый формат видео: {file_ext}")
    
    return True


def generate_output_path(input_path, output_dir=None):
    """Генерирует путь для выходного файла"""
    input_path = Path(input_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_colorized{input_path.suffix}"
    else:
        output_path = input_path.parent / f"{input_path.stem}_colorized{input_path.suffix}"
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Колоризация черно-белого видео с помощью глубокого обучения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py input.mp4
  python main.py input.mp4 -o colorized_output.mp4
  python main.py input.mp4 -o ./results/ --model weights.pth
  python main.py input.mp4 --cpu
        """
    )
    
    parser.add_argument(
        'input',
        help='Путь к входному черно-белому видео'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Путь к выходному файлу или директории (по умолчанию: рядом с входным файлом с суффиксом _colorized)'
    )
    
    parser.add_argument(
        '--model',
        help='Путь к предобученной модели (опционально)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Принудительно использовать CPU вместо GPU'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Обработать только первые 100 кадров для предварительного просмотра'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод'
    )
    
    args = parser.parse_args()
    
    try:
        # Проверяем входной файл
        validate_input_file(args.input)
        
        # Определяем выходной путь
        if args.output:
            if os.path.isdir(args.output):
                output_path = generate_output_path(args.input, args.output)
            else:
                output_path = args.output
        else:
            output_path = generate_output_path(args.input)
        
        # Определяем устройство
        if args.cpu:
            device = 'cpu'
        else:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 60)
        print("🎨 ПРОГРАММА КОЛОРИЗАЦИИ ВИДЕО")
        print("=" * 60)
        print(f"📁 Входной файл: {args.input}")
        print(f"📁 Выходной файл: {output_path}")
        print(f"🖥️  Устройство: {device.upper()}")
        if args.model:
            print(f"🧠 Модель: {args.model}")
        if args.preview:
            print("👁️  Режим предварительного просмотра (100 кадров)")
        print("=" * 60)
        
        # Создаем колоризатор
        print("🔧 Инициализация модели...")
        colorizer = VideoColorizer(model_path=args.model, device=device)
        
        # Засекаем время
        start_time = time.time()
        
        # Колоризуем видео
        print("🎬 Начинаем колоризацию...")
        
        if args.preview:
            # Для предварительного просмотра создаем временную версию
            temp_colorizer = PreviewColorizer(colorizer, max_frames=100)
            temp_colorizer.colorize_video(args.input, output_path, progress_callback)
        else:
            colorizer.colorize_video(args.input, output_path, progress_callback)
        
        # Вычисляем время выполнения
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Колоризация завершена успешно!")
        print(f"⏱️  Время выполнения: {duration:.2f} секунд")
        print(f"📁 Результат сохранен: {output_path}")
        
        # Показываем размер файлов
        input_size = os.path.getsize(args.input) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"📊 Размер входного файла: {input_size:.2f} MB")
        print(f"📊 Размер выходного файла: {output_size:.2f} MB")
        
    except KeyboardInterrupt:
        print("\n❌ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


class PreviewColorizer:
    """Обертка для колоризации только части видео"""
    
    def __init__(self, colorizer, max_frames=100):
        self.colorizer = colorizer
        self.max_frames = max_frames
    
    def colorize_video(self, input_path, output_path, progress_callback=None):
        import cv2
        
        # Открываем входное видео
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {input_path}")
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.max_frames)
        
        # Настраиваем кодек для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"🔍 Предварительный просмотр: {total_frames} кадров")
        
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Колоризуем кадр
            colorized_frame = self.colorizer.colorize_frame(frame)
            
            # Записываем результат
            out.write(colorized_frame)
            
            frame_count += 1
            
            # Обновляем прогресс
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        # Закрываем файлы
        cap.release()
        out.release()


if __name__ == "__main__":
    main()