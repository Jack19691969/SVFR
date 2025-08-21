import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from skimage import color
import cv2


class ColorizationNet(nn.Module):
    """
    Упрощенная модель для колоризации изображений
    Основана на архитектуре U-Net с предобученным энкодером
    """
    
    def __init__(self):
        super(ColorizationNet, self).__init__()
        
        # Энкодер (понижающий путь)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Центральная часть
        self.center = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)
        )
        
        # Декодер (повышающий путь)
        self.up4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2)
        )
        
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # Выходные каналы a и b для Lab цветового пространства
        )
    
    def forward(self, x):
        # Энкодер
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        
        # Центр
        center = self.center(conv4)
        
        # Декодер с skip connections
        up4 = self.up4(torch.cat([center, conv4], 1))
        up3 = self.up3(torch.cat([up4, conv3], 1))
        up2 = self.up2(torch.cat([up3, conv2], 1))
        up1 = self.up1(torch.cat([up2, conv1], 1))
        
        return torch.tanh(up1)  # Ограничиваем выход в диапазоне [-1, 1]


class VideoColorizer:
    """
    Класс для колоризации видео
    """
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ColorizationNet().to(device)
        
        # Если есть предобученная модель, загружаем её
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Загружена предобученная модель: {model_path}")
            except:
                print("Не удалось загрузить предобученную модель. Используем случайные веса.")
        else:
            print("Используем модель со случайными весами (для демонстрации)")
        
        self.model.eval()
        
        # Трансформации для предобработки
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def preprocess_frame(self, frame):
        """
        Предобработка кадра: конвертация в Lab цветовое пространство
        """
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в Lab цветовое пространство
        lab = color.rgb2lab(frame_rgb)
        
        # Нормализуем L канал (яркость) в диапазон [0, 1]
        L = lab[:, :, 0] / 100.0
        
        # Изменяем размер и добавляем batch dimension
        L_resized = cv2.resize(L, (256, 256))
        L_tensor = torch.from_numpy(L_resized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        return L_tensor, lab.shape[:2]
    
    def postprocess_frame(self, L_tensor, ab_tensor, original_shape):
        """
        Постобработка: объединение L и ab каналов в цветное изображение
        """
        # Получаем предсказанные ab каналы
        ab = ab_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        L = L_tensor.squeeze().cpu().numpy()
        
        # Изменяем размер обратно к оригинальному
        ab_resized = cv2.resize(ab, (original_shape[1], original_shape[0]))
        L_resized = cv2.resize(L, (original_shape[1], original_shape[0]))
        
        # Денормализуем L канал
        L_resized = L_resized * 100.0
        
        # Масштабируем ab каналы
        ab_resized = ab_resized * 128.0
        
        # Объединяем каналы
        lab = np.zeros((original_shape[0], original_shape[1], 3))
        lab[:, :, 0] = L_resized
        lab[:, :, 1:] = ab_resized
        
        # Конвертируем обратно в RGB
        rgb = color.lab2rgb(lab)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Конвертируем в BGR для OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def colorize_frame(self, frame):
        """
        Колоризация одного кадра
        """
        with torch.no_grad():
            # Предобработка
            L_tensor, original_shape = self.preprocess_frame(frame)
            
            # Получаем предсказание модели
            ab_pred = self.model(L_tensor)
            
            # Постобработка
            colorized_frame = self.postprocess_frame(L_tensor, ab_pred, original_shape)
            
            return colorized_frame
    
    def colorize_video(self, input_path, output_path, progress_callback=None):
        """
        Колоризация всего видео
        """
        # Открываем входное видео
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {input_path}")
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Настраиваем кодек для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"Обрабатываем видео: {total_frames} кадров, {fps} FPS")
        print(f"Разрешение: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Колоризуем кадр
            colorized_frame = self.colorize_frame(frame)
            
            # Записываем результат
            out.write(colorized_frame)
            
            frame_count += 1
            
            # Обновляем прогресс
            if progress_callback:
                progress_callback(frame_count, total_frames)
            else:
                if frame_count % 30 == 0:  # Выводим прогресс каждые 30 кадров
                    progress = (frame_count / total_frames) * 100
                    print(f"Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Закрываем файлы
        cap.release()
        out.release()
        
        print(f"Колоризация завершена! Результат сохранен в: {output_path}")


def create_demo_weights():
    """
    Создает демонстрационные веса для модели
    (в реальном проекте здесь была бы загрузка предобученной модели)
    """
    model = ColorizationNet()
    
    # Инициализируем веса небольшими случайными значениями
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_normal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    return model.state_dict()


if __name__ == "__main__":
    # Создаем экземпляр колоризатора
    colorizer = VideoColorizer()
    
    # Пример использования
    print("Модель для колоризации видео готова к использованию!")
    print("Используйте main.py для колоризации ваших видео.")