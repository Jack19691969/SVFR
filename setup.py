#!/usr/bin/env python3
"""
Скрипт установки для программы колоризации видео
Автоматически настраивает окружение и устанавливает зависимости
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


def run_command(command, description=""):
    """Выполняет команду и обрабатывает ошибки"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - выполнено")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при {description.lower()}:")
        print(f"   {e.stderr}")
        return False


def check_python_version():
    """Проверяет версию Python"""
    version = sys.version_info
    print(f"🐍 Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или новее")
        return False
    
    print("✅ Версия Python подходит")
    return True


def check_system():
    """Проверяет систему"""
    system = platform.system()
    print(f"💻 Операционная система: {system}")
    
    # Проверяем наличие pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip доступен")
    except subprocess.CalledProcessError:
        print("❌ pip недоступен")
        return False
    
    return True


def install_dependencies():
    """Устанавливает зависимости"""
    print("📦 Установка зависимостей...")
    
    # Обновляем pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Обновление pip"):
        return False
    
    # Устанавливаем основные зависимости
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Установка основных зависимостей"):
        return False
    
    return True


def setup_gpu_support():
    """Настраивает поддержку GPU"""
    print("🚀 Настройка GPU поддержки...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA уже доступна")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA версия: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA недоступна, будет использоваться CPU")
    except ImportError:
        print("⚠️  PyTorch не установлен")
    
    # Предлагаем установить PyTorch с CUDA
    response = input("Установить PyTorch с поддержкой CUDA? (y/N): ").lower()
    
    if response == 'y':
        cuda_command = (f"{sys.executable} -m pip install torch torchvision "
                       "--index-url https://download.pytorch.org/whl/cu118")
        
        if run_command(cuda_command, "Установка PyTorch с CUDA"):
            print("✅ PyTorch с CUDA установлен")
            return True
        else:
            print("⚠️  Не удалось установить PyTorch с CUDA, используйте CPU версию")
    
    return True


def create_directories():
    """Создает необходимые директории"""
    dirs = ["results", "models", "temp"]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана директория: {dir_name}")
        else:
            print(f"📁 Директория уже существует: {dir_name}")


def verify_installation():
    """Проверяет установку"""
    print("🔍 Проверка установки...")
    
    # Проверяем импорты
    modules_to_check = [
        "torch", "torchvision", "cv2", "numpy", 
        "PIL", "skimage", "tqdm", "matplotlib"
    ]
    
    failed_imports = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Не удалось импортировать: {', '.join(failed_imports)}")
        return False
    
    # Проверяем основные файлы
    required_files = ["main.py", "colorization_model.py", "utils.py", "requirements.txt"]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            return False
    
    return True


def run_demo_test():
    """Запускает демонстрационный тест"""
    response = input("\nЗапустить демонстрационный тест? (y/N): ").lower()
    
    if response == 'y':
        print("🎬 Запуск демонстрации...")
        if run_command(f"{sys.executable} demo.py", "Демонстрационный тест"):
            print("✅ Демонстрация успешно завершена!")
        else:
            print("⚠️  Демонстрация завершилась с ошибками")


def main():
    print("🎨 УСТАНОВКА ПРОГРАММЫ КОЛОРИЗАЦИИ ВИДЕО")
    print("=" * 50)
    
    # Проверка системы
    if not check_python_version():
        sys.exit(1)
    
    if not check_system():
        sys.exit(1)
    
    # Установка
    if not install_dependencies():
        print("❌ Не удалось установить зависимости")
        sys.exit(1)
    
    # Настройка GPU
    setup_gpu_support()
    
    # Создание директорий
    create_directories()
    
    # Проверка установки
    if not verify_installation():
        print("❌ Установка завершилась с ошибками")
        sys.exit(1)
    
    print("\n🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 50)
    
    # Показываем информацию о системе
    try:
        from utils import print_system_info
        print_system_info()
    except ImportError:
        print("⚠️  Не удалось загрузить информацию о системе")
    
    print("\n📖 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Запустите: python main.py your_video.mp4")
    print("2. Для помощи: python main.py --help")
    print("3. Для демонстрации: python demo.py")
    
    # Предлагаем запустить демонстрацию
    run_demo_test()


if __name__ == "__main__":
    main()