from uuid import uuid4

import numpy as np
from catboost import CatBoostClassifier
import mne

from utils import extract_features, extract_segments


def predict_states(file_path, model_path='catboost_model.cbm', save_predictions=True):
    """
    Предсказывает состояния для каждого сегмента в файле ЭКоГ и сохраняет результаты в EDF.
    
    Args:
        file_path: путь к EDF файлу
        model_path: путь к сохраненной модели CatBoost
        save_predictions: сохранять ли предсказания в EDF файл
    
    Returns:
        predictions: numpy array с предсказанными состояниями
        timestamps: numpy array с временными метками для каждого предсказания
    """
    # Загружаем модель
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Загружаем данные и информацию о файле
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()
    segments = extract_segments(data)
    features_list = extract_features(segments)
    predictions = model.predict(features_list).reshape(-1)
    predictions_formatted = []

    print("\nPredicted states:")
    previous_pred = 0
    buffer = [0]
    for indx in range(len(predictions)):

        if predictions[indx] == buffer[-1]:
            buffer.append(predictions[indx])
        else:
            if len(buffer) > 5:
                previous_pred = buffer[-1]
                predictions_formatted.extend(buffer)
            else:
                predictions_formatted.extend([previous_pred] * len(buffer))

            buffer = [predictions[indx]]

    if len(buffer) > 5:
        predictions_formatted.extend(buffer)
    else:
        predictions_formatted.extend([previous_pred] * len(buffer))

    predictions = np.array(predictions_formatted[1:]).reshape(-1, 1)

    timestamps = np.arange(len(predictions))

    if save_predictions:
        # Создаем списки для аннотаций
        onset = []
        duration = []
        description = []

        current_state = predictions[0][0]
        start_time = 0

        state_names = {1: 'swd', 2: 'ds', 3: 'is'}

        # Проходим по всем предсказаниям и создаем аннотации
        for i in range(1, len(predictions)):
            if predictions[i][0] != current_state or i == len(predictions) - 1:
                # Добавляем две аннотации для каждого состояния (начало и конец)
                if current_state != 0:  # Пропускаем фоновое состояние
                    state_name = state_names[current_state]

                    # Начало состояния
                    onset.append(float(start_time))
                    duration.append(0.0)
                    description.append(f"{state_name}1")

                    # Конец состояния
                    onset.append(float(i))
                    duration.append(0.0)
                    description.append(f"{state_name}2")

                    print(f"{state_name}1", start_time)
                    print(f"{state_name}2", i)

                start_time = i
                current_state = predictions[i][0]

        # Создаем объект аннотаций
        annotations = mne.Annotations(
            onset=onset,
            duration=duration,
            description=description
        )

        # Создаем новый Raw объект с предсказаниями
        raw_predicted = raw.copy()
        raw_predicted.set_annotations(annotations)

        # Формируем имя выходного файла, сохраняя в той же директории
        output_path = f'{uuid4()}_predicted.edf'

        # Сохраняем как EDF файл
        mne.export.export_raw(output_path, raw_predicted, fmt='edf', overwrite=True)
        print(f"Predictions saved to: {output_path}")

    return predictions.reshape(-1), timestamps


def get_state_name(state_id):
    """Преобразует числовой идентификатор состояния в текстовое описание."""
    states = {
        0: "Background",
        1: "Spike-wave discharge",
        2: "Delta sleep",
        3: "Intermediate sleep"
    }
    return states.get(state_id, "Unknown")


if __name__ == "__main__":
    # Пример использования
    file_path = "nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x1_15m_BL_6h.edf"
    predictions, timestamps = predict_states(file_path)

    # Выводим результаты
    print("\nPredicted states:")
    for indx in range(len(predictions)):
        minutes = int(timestamps[indx] // 60)
        seconds = int(timestamps[indx] % 60)

        state_name = get_state_name(predictions[indx])
        previous_pred = predictions[indx]

        print(f"Time {minutes:02d}:{seconds:02d} - {state_name}")
