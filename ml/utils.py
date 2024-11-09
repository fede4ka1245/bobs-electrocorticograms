from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy import stats
import pywt
from tqdm import tqdm

import mne
import pandas as pd


def different_val_train_paths(train_paths, val_paths, use_extract_features=True):
    segments, labels = create_dataset(train_paths)

    print("Label distribution before train:", Counter(labels))

    # drop some segments where label == 0
    for index_to_shrink in [0, 1, 2, 3]:
        idx = np.where(labels == index_to_shrink)[0]
        np.random.seed(42)
        drop_idx = np.random.choice(idx, size=max(0, len(idx) - 1000), replace=False)
        segments = np.delete(segments, drop_idx, axis=0)
        labels = np.delete(labels, drop_idx)

    print("Label distribution after train:", Counter(labels))

    print("Extracting features train...")

    if use_extract_features:
        features = extract_features(segments)

        print(f"Extracted {features.shape[1]} features per segment train")

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(features, labels)
    else:
        X_train_smote, y_train_smote = segments, labels

    print("Label distribution after smote:", Counter(y_train_smote))

    print('Extracting features val...')

    segments, labels = create_dataset(val_paths)

    if use_extract_features:
        features = extract_features(segments)
        X_val, y_val = features, labels
    else:
        X_val, y_val = segments, labels

    print('Shapes', X_train_smote.shape, X_val.shape, y_train_smote.shape, y_val.shape)

    return X_train_smote, X_val, y_train_smote, y_val


def load_edf_file(file_path):
    """Load EDF file and extract data and annotations."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    annotations = raw.annotations
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    return data, annotations


def extract_segments(data, annotations=None, sampling_rate=400):
    """Extract one-second segments labeled with SWD, DS, or IS from EDF data."""

    # print(data.shape)

    segments = []

    for i in range(0, data.shape[1], sampling_rate):
        segment = data[:, max(0, i - 1 * sampling_rate):min(data.shape[1], i + 2 * sampling_rate)]

        left_pad = (sampling_rate * 3 - segment.shape[1]) // 2
        right_pad = sampling_rate * 3 - segment.shape[1] - left_pad
        segment = np.pad(segment, ((0, 0), (left_pad, right_pad)), mode='constant')

        segments.append(segment)

    # segments = data.reshape(-1, 3, sampling_rate)

    if annotations is None:
        return np.array(segments, dtype=np.float32)

    labels = np.zeros(len(segments))
    annotation_mapping = {'swd': 1, 'ds': 2, 'is': 3}  # Assign numerical labels

    for annot_index in range(1, len(annotations)):
        prev_label_key = annotations[annot_index - 1]['description']
        label_key = annotations[annot_index]['description']

        if prev_label_key[-1] == '1' and label_key[-1] == '2' and \
                prev_label_key[:-1] == label_key[:-1]:
            start = int(annotations[annot_index - 1]['onset'])
            end = int(annotations[annot_index]['onset'])

            labels[start:end] = annotation_mapping[label_key[:-1]]

    # Convert to numpy arrays
    segments = np.array(segments, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(segments.shape)

    return segments, labels


def create_dataset(file_paths, sampling_rate=400):
    """Load multiple EDF files and create a dataset of one-second labeled segments."""
    all_segments = []
    all_labels = []

    for file_path in file_paths:
        data, annotations = load_edf_file(file_path)
        segments, labels = extract_segments(data, annotations, sampling_rate)
        all_segments.append(segments)
        all_labels.append(labels)

    # Combine all segments and labels from different files
    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_segments, all_labels


def extract_features(segments):
    """
    Извлекает доменные признаки из сегментов ЭКоГ.
    """
    n_segments = segments.shape[0]
    features_list = []

    # Нормализация всего датасета с проверкой на нулевую дисперсию
    std = np.std(segments)
    if std < 1e-10:
        segments = (segments - np.mean(segments))  # Только центрируем данные
    else:
        segments = (segments - np.mean(segments)) / std

    for i in tqdm(range(n_segments)):
        segment_features = []

        for channel in range(segments.shape[1]):
            signal = segments[i, channel]

            # Проверка на одинаковые значения
            if np.all(signal == signal[0]):
                # Если все значения одинаковые, пропускаем статистические вычисления
                segment_features.extend([0.0] * 30)  # для лагов
                segment_features.extend([signal[0], 0.0, 0.0, 0.0, signal[0], signal[0], signal[0], 0.0, 0.0, 0.0])
            else:
                # Добавляем лаги для текущего канала
                for lag in range(1, 31):
                    lagged_signal = np.roll(signal, lag)
                    segment_features.append(lagged_signal[0])

                # Статистические характеристики во временной области
                segment_features.extend([
                    np.mean(signal),
                    np.std(signal) + 1e-10,
                    float(skew(signal, bias=True)),
                    float(kurtosis(signal, bias=True)),
                    np.max(signal),
                    np.min(signal),
                    np.median(signal),
                    float(stats.iqr(signal)),
                    np.mean(np.abs(signal)),
                    np.sum(np.abs(np.diff(signal)))
                ])

            # Частотные характеристики с проверкой на нулевой сигнал
            if np.all(np.abs(signal) < 1e-10):
                segment_features.extend([0.0] * 12)  # для частотных характеристик
            else:
                freqs, psd = welch(signal, fs=400, nperseg=min(256, len(signal)))
                eps = 1e-10

                # Проверка на нулевые значения в диапазонах
                delta_power = max(np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]), eps)
                theta_power = max(np.sum(psd[(freqs >= 4) & (freqs <= 8)]), eps)
                alpha_power = max(np.sum(psd[(freqs >= 8) & (freqs <= 13)]), eps)
                beta_power = max(np.sum(psd[(freqs >= 13) & (freqs <= 30)]), eps)
                gamma_power = max(np.sum(psd[(freqs >= 30) & (freqs <= 100)]), eps)

                segment_features.extend([
                    np.log1p(delta_power),
                    np.log1p(theta_power),
                    np.log1p(alpha_power),
                    np.log1p(beta_power),
                    np.log1p(gamma_power),
                    theta_power / delta_power,
                    alpha_power / delta_power,
                    beta_power / delta_power,
                    np.log1p(np.max(psd)),
                    freqs[np.argmax(psd)],
                    np.log1p(np.sum(psd)),
                    np.log1p(np.mean(psd))
                ])

            # Вейвлет-преобразование с обработкой ошибок
            try:
                if np.all(np.abs(signal) < 1e-10):
                    segment_features.extend([0.0] * (3 * 5))
                else:
                    coeffs = pywt.wavedec(signal, 'db4', level=4)
                    for coeff in coeffs:
                        if len(coeff) > 0:
                            segment_features.extend([
                                np.mean(np.abs(coeff)),
                                np.std(coeff) + eps,
                                np.max(np.abs(coeff))
                            ])
                        else:
                            segment_features.extend([0.0, 0.0, 0.0])
            except Exception as e:
                print(f"Warning: Wavelet transform failed: {e}")
                segment_features.extend([0.0] * (3 * 5))

            # Нелинейные характеристики с проверкой на нулевой сигнал
            if np.all(np.abs(signal) < 1e-10):
                segment_features.extend([0.0] * 5)
            else:
                zero_crossings = np.sum(np.diff(np.signbit(signal).astype(int)))
                diff1 = np.diff(signal)
                diff2 = np.diff(diff1)

                std_signal = max(np.std(signal), eps)
                std_diff1 = max(np.std(diff1), eps)
                std_diff2 = max(np.std(diff2), eps)

                segment_features.extend([
                    zero_crossings,
                    std_diff1 / std_signal,
                    (std_diff2 * std_signal) / (std_diff1 * std_diff1),
                    np.mean(np.abs(diff1)),
                    np.mean(np.abs(diff2))
                ])

            # Характеристики пиков с проверкой на минимальное количество точек
            peak_indices = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
            if len(peak_indices) > 1:
                peak_intervals = np.diff(peak_indices)
                segment_features.extend([
                    np.mean(peak_intervals),
                    np.std(peak_intervals) + eps,
                    len(peak_indices) / len(signal) * 400
                ])
            else:
                segment_features.extend([0.0, 0.0, 0.0])

        features_list.append(segment_features)

    features = np.array(features_list)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Дополнительная проверка на выбросы
    features = np.clip(features, -1e6, 1e6)

    return features


def balance_data_and_split(file_paths, use_extract_features=True):
    segments, labels = create_dataset(file_paths)

    print("Label distribution before:", Counter(labels))

    # drop some segments where label == 0
    idx = np.where(labels == 0)[0]
    np.random.seed(42)
    drop_idx = np.random.choice(idx, size=len(idx) - 700, replace=False)
    segments = np.delete(segments, drop_idx, axis=0)
    labels = np.delete(labels, drop_idx)

    print("Label distribution after:", Counter(labels))

    if use_extract_features:
        print("Extracting features...")
        features = extract_features(segments)

        print(f"Extracted {features.shape} features per segment")
    else:
        features = segments

    print(features.shape)

    np.save("features.npy", features)
    np.save("labels.npy", labels)

    # Балансируем только тренировочные данные
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    if use_extract_features:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print("Label distribution after smote:", Counter(y_train_smote))
        return X_train_smote, X_val, y_train_smote, y_val

    return X_train, X_val, y_train, y_val


def create_features_dataframe(features):
    """
    Создает pandas DataFrame с описанием всех признаков.

    Args:
        features: numpy array с признаками
    Returns:
        pandas DataFrame с названиями и описаниями признаков
    """
    feature_names = []
    feature_descriptions = []

    for channel in range(3):
        channel_name = f'Channel_{channel + 1}'

        # Лаговые признаки
        for lag in range(1, 31):
            feature_names.append(f'{channel_name}_lag_{lag}')
            feature_descriptions.append(f'Значение сигнала со сдвигом {lag} для канала {channel + 1}')

        # Статистические характеристики
        stats_features = [
            ('mean', 'Среднее значение сигнала'),
            ('std', 'Стандартное отклонение'),
            ('skew', 'Коэффициент асимметрии'),
            ('kurtosis', 'Коэффициент эксцесса'),
            ('max', 'Максимальное значение'),
            ('min', 'Минимальное значение'),
            ('median', 'Медиана'),
            ('iqr', 'Межквартильный размах'),
            ('abs_mean', 'Среднее абсолютное значение'),
            ('diff_sum', 'Сумма абсолютных разностей')
        ]

        for name, desc in stats_features:
            feature_names.append(f'{channel_name}_{name}')
            feature_descriptions.append(desc)

        # Частотные характеристики
        freq_features = [
            ('delta_power', 'Мощность в дельта-диапазоне (0.5-4 Гц)'),
            ('theta_power', 'Мощность в тета-диапазоне (4-8 Гц)'),
            ('alpha_power', 'Мощность в альфа-диапазоне (8-13 Гц)'),
            ('beta_power', 'Мощность в бета-диапазоне (13-30 Гц)'),
            ('gamma_power', 'Мощность в гамма-диапазоне (30-100 Гц)'),
            ('theta_delta_ratio', 'Отношение тета/дельта мощностей'),
            ('alpha_delta_ratio', 'Отношение альфа/дельта мощностей'),
            ('beta_delta_ratio', 'Отношение бета/дельта мощностей'),
            ('peak_power', 'Пиковая мощность'),
            ('peak_freq', 'Частота пика мощности'),
            ('total_power', 'Общая мощность'),
            ('mean_power', 'Средняя мощность')
        ]

        for name, desc in freq_features:
            feature_names.append(f'{channel_name}_{name}')
            feature_descriptions.append(desc)

        # Вейвлет-характеристики
        for level in range(5):
            wavelet_features = [
                ('mean_abs', f'Среднее абсолютное значение коэффициентов уровня {level}'),
                ('std', f'Стандартное отклонение коэффициентов уровня {level}'),
                ('max_abs', f'Максимальное абсолютное значение коэффициентов уровня {level}')
            ]

            for name, desc in wavelet_features:
                feature_names.append(f'{channel_name}_wavelet_level{level}_{name}')
                feature_descriptions.append(desc)

        # Нелинейные характеристики
        nonlinear_features = [
            ('zero_crossings', 'Количество пересечений нуля'),
            ('mobility', 'Мобильность сигнала'),
            ('complexity', 'Сложность сигнала'),
            ('diff1_mean', 'Среднее первой производной'),
            ('diff2_mean', 'Среднее второй производной')
        ]

        for name, desc in nonlinear_features:
            feature_names.append(f'{channel_name}_{name}')
            feature_descriptions.append(desc)

        # Характеристики пиков
        peak_features = [
            ('peak_intervals_mean', 'Среднее расстояние между пиками'),
            ('peak_intervals_std', 'Стандартное отклонение расстояний между пиками'),
            ('peak_frequency', 'Частота появления пиков')
        ]

        for name, desc in peak_features:
            feature_names.append(f'{channel_name}_{name}')
            feature_descriptions.append(desc)

    # Создаем DataFrame
    features_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Description': feature_descriptions,
        'Index': range(len(feature_names))
    })

    return features_df


def save_feature_importance(feature_importance, features_df, output_file='feature_importance.csv'):
    """
    Сохраняет важность признаков в CSV файл с их описаниями.

    Args:
        feature_importance: массив важностей признаков
        features_df: DataFrame с описанием признаков
        output_file: путь для сохранения результата
    """
    importance_df = features_df.copy()
    importance_df['Importance'] = feature_importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df.to_csv(output_file, index=False)

    return importance_df
