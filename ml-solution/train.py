import glob
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
from scipy.signal import resample
import math

from utils import different_val_train_paths


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class ECoGClassifier(nn.Module):
    def __init__(self, input_channels=3, d_model=128, nhead=8, num_layers=3, dropout_rate=0.3):
        super(ECoGClassifier, self).__init__()

        # Расширенная входная обработка для лучшего захвата паттернов
        self.input_conv = nn.Sequential(
            # Первый блок для локальных паттернов
            nn.Conv1d(input_channels, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Второй блок для веретен сна (12-14 Гц)
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=25, padding=12),  # ~25ms для захвата веретен
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Третий блок для K-комплексов
            nn.Conv1d(d_model // 2, d_model, kernel_size=51, padding=25),  # ~50ms для K-комплексов
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.MaxPool1d(2)
        )

        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model)

        # Многоголовый self-attention для разных частотных диапазонов
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Специализированный attention для разных состояний сна
        self.sleep_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1),
                nn.Softmax(dim=1)
            ) for _ in range(4)
        ])

        # Классификатор с дополнительными признаками
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 4)
        )

    def forward(self, x):
        # Входная обработка
        x = self.input_conv(x)
        x = x.transpose(1, 2)

        # Позиционное кодирование
        x = x * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)

        # Transformer
        transformer_output = self.transformer_encoder(x)

        # Применяем специализированные attention для каждого состояния
        context_vectors = []
        for attention in self.sleep_attention:
            weights = attention(transformer_output)
            context = torch.sum(weights * transformer_output, dim=1)
            context_vectors.append(context)

        # Конкатенируем все контекстные векторы
        combined_context = torch.cat(context_vectors, dim=1)

        # Классификация
        output = self.classifier(combined_context)

        return output


# Функция для обучения модели
def train_model(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'mps')
    model = model.to(device)

    # Взвешенная функция потерь для решения проблемы дисбаланса
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Циклический learning rate с warmup
    def warmup_cosine_schedule(epoch):
        if epoch < 10:  # warmup
            return epoch / 10
        return 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (num_epochs - 10)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)

    best_val_f1 = 0.0

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/ecog_classifier_{current_time}')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels_list = []

        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)

            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels_list.extend(batch_labels.cpu().numpy())

        train_f1 = f1_score(train_labels_list, train_predictions, average='macro')
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())

        val_f1 = f1_score(val_labels_list, val_predictions, average='macro')
        avg_val_loss = val_loss / len(val_loader)

        # Добавляем вычисление F1 для каждого класса
        val_f1_per_class = f1_score(val_labels_list, val_predictions, average=None)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('F1-score/train', train_f1, epoch)
        writer.add_scalar('F1-score/validation', val_f1, epoch)

        # Логируем F1 для каждого класса
        writer.add_scalar('F1-score/class_0', val_f1_per_class[0], epoch)
        writer.add_scalar('F1-score/class_1', val_f1_per_class[1], epoch)
        writer.add_scalar('F1-score/class_2', val_f1_per_class[2], epoch)
        # writer.add_scalar('F1-score/class_3', val_f1_per_class[3], epoch)

        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}')
        print('Val F1 per class:')
        print(f'  Background: {val_f1_per_class[0]:.4f}')
        print(f'  Spike-wave discharge: {val_f1_per_class[1]:.4f}')
        print(f'  Delta sleep: {val_f1_per_class[2]:.4f}')
        print(f'  Intermediate sleep: {val_f1_per_class[3]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

        # Сохраняем модель по лучшему F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_f1': train_f1,
                'val_f1': val_f1,
            }, 'best_model.pth')

        scheduler.step()

    writer.close()


class ECoGDataset(Dataset):
    def __init__(self, segments, labels, transform=None):
        self.segments = torch.FloatTensor(segments)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]

        if self.transform:
            segment_np = segment.numpy()
            segment_np = self.transform(segment_np)
            segment = torch.FloatTensor(segment_np)

        return segment, label


class TimeSeriesAugmentation:
    def __init__(self, p=0.5):
        """
        Args:
            p (float): вероятность применения каждой аугментации
        """
        self.p = p

    def gaussian_noise(self, data, noise_factor=0.05):
        """Добавление гауссовского шума"""
        noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
        return data + noise

    def time_shift(self, data, max_shift=20):
        """Случайный сдвиг по времени"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(data, shift, axis=1)

    def amplitude_scale(self, data, factor_range=(0.7, 1.3)):
        """Масштабирование амплитуды"""
        factor = np.random.uniform(*factor_range)
        return data * factor

    def time_stretch(self, data, factor_range=(0.8, 1.2)):
        """Растяжение/сжатие по времени"""
        factor = np.random.uniform(*factor_range)
        orig_len = data.shape[1]
        new_len = int(orig_len * factor)

        # Растягиваем/сжимаем
        stretched = np.array([resample(channel, new_len) for channel in data])

        # Возвращаем к исходной длине
        if new_len > orig_len:
            return stretched[:, :orig_len]
        else:
            padded = np.pad(stretched, ((0, 0), (0, orig_len - new_len)), mode='edge')
            return padded

    def random_cutout(self, data, max_cut_size=40):
        """Зануление случайного участка сигнала"""
        cut_size = np.random.randint(10, max_cut_size)
        cut_start = np.random.randint(0, data.shape[1] - cut_size)
        mask = np.ones(data.shape)
        mask[:, cut_start:cut_start + cut_size] = 0
        return data * mask

    def baseline_wander(self, data, max_wander=0.1):
        """Добавление низкочастотного тренда"""
        t = np.linspace(0, 1, data.shape[1])
        wander = max_wander * np.sin(2 * np.pi * np.random.rand() * t)
        return data + wander

    def __call__(self, data):
        """
        Args:
            data (np.ndarray): входные данные формы (channels, time_steps)
        """
        augmented = data.copy()

        # Применяем аугментации с заданной вероятностью
        if random.random() < self.p:
            augmented = self.gaussian_noise(augmented)
        if random.random() < self.p:
            augmented = self.time_shift(augmented)
        if random.random() < self.p:
            augmented = self.amplitude_scale(augmented)
        if random.random() < self.p:
            augmented = self.time_stretch(augmented)
        if random.random() < self.p:
            augmented = self.random_cutout(augmented)
        if random.random() < self.p:
            augmented = self.baseline_wander(augmented)

        return augmented


if __name__ == "__main__":
    # Example usage
    file_paths = glob.glob('nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/*fully_marked*edf')

    print(file_paths)

    X_train, X_val, y_train, y_val = different_val_train_paths(
        file_paths[:4],
        file_paths[4:],
        False
    )

    print(Counter(y_train))
    # print(X_val.shape)

    batch_size = 64

    # Создаем аугментации только для тренировочного датасета
    # train_transform = TimeSeriesAugmentation(p=0.5)

    # Создание датасетов
    train_dataset = ECoGDataset(
        X_train,
        y_train,
        # transform=train_transform
    )

    val_dataset = ECoGDataset(
        X_val,
        y_val
    )

    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print('Data loaded')

    # Создание и обучение модели
    model = ECoGClassifier()
    train_model(model, train_loader, val_loader, num_epochs=300)
