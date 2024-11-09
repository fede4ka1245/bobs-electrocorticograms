import glob

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import Counter

from utils import create_features_dataframe, save_feature_importance, different_val_train_paths


# Загрузка данных
def load_and_prepare_data():
    file_paths = glob.glob('nmic_ivnd_dataset/ECoG_fully_marked_(4+2 files, 6 h each)/*fully_marked*edf')
    X_train_smote, X_val, y_train_smote, y_val = different_val_train_paths(
        file_paths[:4],
        file_paths[2:]
    )

    return X_train_smote, X_val, y_train_smote, y_val


def train_catboost(X_train, X_val, y_train, y_val):
    # Настраиваем модель с учетом дисбаланса
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=5,
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        task_type='GPU' if GPU_AVAILABLE else 'CPU',
        auto_class_weights='Balanced'
    )

    # Обучаем модель
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=True
    )

    return model


def evaluate_model(model, X_val, y_val):
    # Получаем предсказания
    y_pred = model.predict(X_val)

    # Вычисляем и выводим метрики с zero_division=0
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    print(f'\nMacro F1 Score: {f1:.4f}')

    # Выводим метрики по каждому классу
    print('\nClassification Report:')
    print(classification_report(y_val, y_pred, zero_division=0))

    # Создаем и сохраняем confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Нормализуем confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Создаем два графика: для абсолютных и нормализованных значений
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Absolute)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_xticklabels(['Background', 'SWD', 'Delta Sleep', 'Int. Sleep'])  # , 'Int. Sleep'
    ax1.set_yticklabels(['Background', 'SWD', 'Delta Sleep', 'Int. Sleep'])

    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_xticklabels(['Background', 'SWD', 'Delta Sleep', 'Int. Sleep'])
    ax2.set_yticklabels(['Background', 'SWD', 'Delta Sleep', 'Int. Sleep'])

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Выводим распределение предсказаний
    print("\nPredictions distribution:")
    print(Counter(y_pred.reshape(-1)))

    # Получаем важность признаков
    feature_importance = model.get_feature_importance()

    # Создаем DataFrame с описанием признаков
    features_df = create_features_dataframe(X_val)

    # Сохраняем важность признаков с описаниями
    importance_df = save_feature_importance(feature_importance, features_df)

    # Выводим топ-10 самых важных признаков
    print("\nTop 10 most important features:")
    print(importance_df[['Feature_Name', 'Description', 'Importance']].head(10))

    # Визуализируем важность признаков (топ 20)
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), feature_importance[:20])
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    return f1, feature_importance


if __name__ == "__main__":
    # Проверяем доступность GPU
    GPU_AVAILABLE = False
    try:
        import cupy

        GPU_AVAILABLE = True
    except ImportError:
        pass

    print("Loading data...")
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    print("\nTraining CatBoost model...")
    model = train_catboost(X_train, X_val, y_train, y_val)

    print("\nEvaluating model...")
    f1, feature_importance = evaluate_model(model, X_val, y_val)

    # Сохраняем модель
    model.save_model('catboost_model.cbm')

    print("\nModel saved as 'catboost_model.cbm'")
