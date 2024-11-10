import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import os
import plotly.express as px
from scipy import signal
import plotly.subplots as sp
import mne
from plotly.subplots import make_subplots
from ml.catboost_inference import predict_states

class SleepStageAnalyzer:
    """Класс для анализа стадий сна по ЭКоГ данным"""
    
    def __init__(self, model_path: str = "sleep_stage_model.pth"):
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        self.sampling_rate = 250  # Гц
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Загрузка предобученной модели"""
        if os.path.exists(model_path):
            return torch.load(model_path)
        else:
            st.error(f"Модель не найдена по пути: {model_path}")
            return None

    def preprocess_data(self, raw_data: np.ndarray) -> np.ndarray:
        """Предобработка ЭКоГ сигнала"""
        # Фильтрация сигнала
        nyquist = self.sampling_rate * 0.5
        b, a = butter(4, [0.5/nyquist, 30/nyquist], btype='band')
        filtered_data = filtfilt(b, a, raw_data)
        
        # Нормализация
        normalized_data = self.scaler.fit_transform(filtered_data.reshape(-1, 1))
        return normalized_data
    
    def detect_sleep_stages(self, data: np.ndarray, window_size: int = 30) -> Dict:
        """Определение стадий сна"""
        preprocessed_data = self.preprocess_data(data)
        predictions = self.model(torch.FloatTensor(preprocessed_data))
        
        return {
            'deep_sleep': (predictions == 0).float().mean().item(),
            'intermediate': (predictions == 1).float().mean().item(),
            'wake': (predictions == 2).float().mean().item()
        }

    def create_label_studio_view(self, data: np.ndarray, window_size: int = 7500):
        """Создает визуализацию в стиле Label Studio"""
        # Создаем временную ось
        time = np.arange(len(data)) / self.sampling_rate
        
        # Создаем фигуру с двумя подграфиками
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Сигнал ЭКоГ", "Разметка")
        )
        
        # Добавляем основной сигнал
        fig.add_trace(
            go.Scatter(
                x=time,
                y=data,
                name="ЭКоГ",
                line=dict(color='#2196F3'),
                hovertemplate='Время: %{x:.2f}с<br>Значение: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Добавляем область для разметки
        fig.add_trace(
            go.Scatter(
                x=time,
                y=np.zeros_like(time),
                name="Разметка",
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fill='tozeroy',
                fillcolor='rgba(0,0,0,0.1)',
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Настраиваем внешний вид
        fig.update_layout(
            height=600,
            showlegend=False,
            dragmode='pan',
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Глубокий сон",
                            method="restyle",
                            args=[{"fillcolor": "rgba(33, 150, 243, 0.3)"}],
                            args2=[{"fillcolor": "rgba(0,0,0,0.1)"}]
                        ),
                        dict(
                            label="Промежуточная фаза",
                            method="restyle",
                            args=[{"fillcolor": "rgba(76, 175, 80, 0.3)"}],
                            args2=[{"fillcolor": "rgba(0,0,0,0.1)"}]
                        ),
                        dict(
                            label="Бодрствование",
                            method="restyle",
                            args=[{"fillcolor": "rgba(244, 67, 54, 0.3)"}],
                            args2=[{"fillcolor": "rgba(0,0,0,0.1)"}]
                        )
                    ],
                    x=0.1,
                    y=1.1,
                    direction="right"
                )
            ]
        )
        
        # Добавляем инструменты навигации
        fig.update_xaxes(
            rangeslider=dict(visible=True),
            row=2, col=1
        )
        
        return fig

def create_web_interface():
    st.set_page_config(page_title="Анализ ЭКоГ сна крыс WAG/Rij", layout="wide")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.title("Настройки анализа")
        window_size = st.slider("Размер окна анализа (сек)", 10, 60, 30)
        filter_type = st.selectbox(
            "Тип фильтрации",
            ["Полосовой фильтр", "Вейвлет-преобразование"]
        )
        
        st.subheader("Дополнительные параметры")
        show_spectogram = st.checkbox("Показать спектрограмму", True)
        show_statistics = st.checkbox("Показать статистику", True)
        
    # Основной интерфейс
    st.title("Анализ фаз сна крыс WAG/Rij")
    
    # Загрузка данных
    uploaded_file = st.file_uploader(
        "Загрузите файл ЭКоГ (.csv, .edf)", 
        type=['csv', 'edf']
    )
    
    if uploaded_file:
        # Создаем объект анализатора
        analyzer = SleepStageAnalyzer()
        
        # Загружаем и обрабатываем данные
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:  # .edf
            # Сохраняем временный файл для работы с MNE
            temp_file = "temp.edf"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Загружаем EDF файл
                raw = mne.io.read_raw_edf(temp_file, preload=True)
                
                # Получаем частоту дискретизации
                analyzer.sampling_rate = raw.info['sfreq']
                
                # Определяем размер чанка (5 минут данных)
                chunk_duration = 300  # секунд
                chunk_samples = int(chunk_duration * analyzer.sampling_rate)
                
                # Позволяем пользователю выбрать временной интервал
                total_duration = len(raw.times) / analyzer.sampling_rate
                start_time = st.slider(
                    "Выберите начало интервала (минуты)",
                    0,
                    int(total_duration/60),
                    0
                ) * 60  # конвертируем в секунды
                
                # Получаем данные из EDF файла
                start_sample = int(start_time * analyzer.sampling_rate)
                raw_data = raw.get_data()
                signal_data = raw_data[0]  # Берем первый канал
                
                # Создаем DataFrame для совместимости с остальным кодом
                data = pd.DataFrame(signal_data.T, columns=['ECoG'])
                
                # Получаем предсказания
                predictions, timestamps = predict_states(signal_data)
                
                st.info(f"""Загружен EDF файл:
                    - Каналов: {len(raw.ch_names)}
                    - Частота дискретизации: {analyzer.sampling_rate} Гц
                    - Общая длительность: {total_duration/60:.1f} минут
                    - Текущий интервал: {start_time/60:.1f}-{(start_time + chunk_duration)/60:.1f} минут
                """)
                
            except Exception as e:
                st.error(f"Ошибка при чтении EDF файла: {str(e)}")
                return
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Создаем вкладки для разных типов визуализации
        tab1, tab2, tab3 = st.tabs(["Разметка", "Анализ", "Статистика"])
        
        with tab1:
            st.subheader("Разметка данных")
            
            # Создаем визуализацию в стиле Label Studio
            label_studio_fig = analyzer.create_label_studio_view(data.iloc[:, 0].values)
            st.plotly_chart(label_studio_fig, use_container_width=True)
            
            # Добавляем элементы управления
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider(
                    "Размер окна (сек)",
                    1,
                    60,
                    30
                )
            with col2:
                st.download_button(
                    "Скачать разметку",
                    data="",  # Здесь будет JSON с разметкой
                    file_name="labels.json",
                    mime="application/json"
                )
        
        with tab2:
            # Визуализация данных
            st.subheader("Визуализация ЭКоГ данных")
            
            # Создаем вкладки для разных типов визуализации
            tab1, tab2, tab3 = st.tabs(["Временной анализ", "Частотный анализ", "Детальный просмотр"])
            
            with tab1:
                # Создаем графк с двумя осями Y
                fig = sp.make_subplots(rows=2, cols=1, 
                                     shared_xaxes=True,
                                     vertical_spacing=0.02,
                                     subplot_titles=("Исходный сигнал ЭКоГ", "Отфильтрованный сигнал"))
                
                # Исходный сигнал
                fig.add_trace(
                    go.Scatter(
                        y=data.iloc[:, 0],
                        name="Исходный сигнал",
                        line=dict(color='#1f77b4')
                    ),
                    row=1, col=1
                )
                
                # Отфильтрованный сигнал
                filtered_data = analyzer.preprocess_data(data.iloc[:, 0].values)
                fig.add_trace(
                    go.Scatter(
                        y=filtered_data.flatten(),
                        name="Отфильтрованный сигнал",
                        line=dict(color='#2ca02c')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Добавляем интерфейс динамического окна под графиком
                st.subheader("Динамическое окно")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Слайдер для выбра временного окна
                    window_position = st.slider(
                        "Положение окна (сек)",
                        0,
                        int(len(data)/analyzer.sampling_rate) - window_size,
                        0
                    )
                
                with col2:
                    # Выбор размера окна
                    window_duration = st.number_input(
                        "Размер окна (сек)",
                        min_value=1,
                        max_value=60,
                        value=10
                    )
                
                with col3:
                    # Кнопка для обновления анализа
                    if st.button("Анализировать окно"):
                        start_idx = int(window_position * analyzer.sampling_rate)
                        end_idx = int(start_idx + window_duration * analyzer.sampling_rate)
                        
                        # График выбранного окна
                        window_fig = go.Figure()
                        window_fig.add_trace(go.Scatter(
                            y=data.iloc[start_idx:end_idx, 0],
                            name="Сигнал в окне",
                            line=dict(color='#d62728')
                        ))
                        
                        window_fig.update_layout(
                            title=f'Выбранное окно (с {window_position} по {window_position + window_duration} сек)',
                            xaxis_title='Отсчеты',
                            yaxis_title='Амплитуда',
                            height=300,
                            showlegend=True
                        )
                        
                        st.plotly_chart(window_fig, use_container_width=True)
                        
                        # Анализ выбранного окна
                        window_data = data.iloc[start_idx:end_idx, 0].values
                        window_stages = analyzer.detect_sleep_stages(window_data)
                        
                        # Показываем результаты анализа окна
                        st.subheader("Анализ выбранного окна")
                        wcol1, wcol2, wcol3 = st.columns(3)
                        with wcol1:
                            st.metric("Глубокий сон", f"{window_stages['deep_sleep']:.1%}")
                        with wcol2:
                            st.metric("Промежуточная фаза", f"{window_stages['intermediate']:.1%}")
                        with wcol3:
                            st.metric("Бодрствование", f"{window_stages['wake']:.1%}")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Спектрограмма
                    st.subheader("Спектрограмма")
                    f, t, Sxx = signal.spectrogram(data.iloc[:, 0], 
                                                 fs=analyzer.sampling_rate,
                                                 nperseg=256,
                                                 noverlap=128)
                    
                    fig_spectro = go.Figure(data=go.Heatmap(
                        z=10 * np.log10(Sxx),
                        x=t,
                        y=f,
                        colorscale='Viridis'
                    ))
                    
                    fig_spectro.update_layout(
                        title='Спектрограма сигнала',
                        xaxis_title='Время (с)',
                        yaxis_title='Частота (Гц)',
                        height=400
                    )
                    st.plotly_chart(fig_spectro, use_container_width=True)
                
                with col2:
                    # График спектральной плотности мощности
                    st.subheader("Спектральная плотность мощности")
                    f, Pxx = signal.welch(data.iloc[:, 0], 
                                        fs=analyzer.sampling_rate,
                                        nperseg=1024)
                    
                    fig_psd = go.Figure(data=go.Scatter(
                        x=f,
                        y=10 * np.log10(Pxx),
                        mode='lines',
                        line=dict(color='#d62728')
                    ))
                    
                    fig_psd.update_layout(
                        title='Спектральная плотность мощности',
                        xaxis_title='Частота (Гц)',
                        yaxis_title='Мощность (дБ/Гц)',
                        height=400
                    )
                    st.plotly_chart(fig_psd, use_container_width=True)
                
                with tab3:
                    # Интерактивный просмотр участков сигнала
                    st.subheader("Детальный просмотр сигнала")
                    
                    # Слайдер для выбора временного окна
                    window_start = st.slider(
                        "Выберите начальную точку (сек)",
                        0,
                        int(len(data)/analyzer.sampling_rate) - window_size,
                        0
                    )
                    
                    # Вычисляем индексы для выбранного окна и преобразуем их в целые числа
                    start_idx = int(window_start * analyzer.sampling_rate)
                    end_idx = int(start_idx + window_size * analyzer.sampling_rate)
                    
                    # Создаем детальный график выбранного участка
                    fig_detailed = go.Figure()
                    
                    fig_detailed.add_trace(go.Scatter(
                        y=data.iloc[start_idx:end_idx, 0],
                        name="Сигнал",
                        line=dict(color='#17becf')
                    ))
                    
                    fig_detailed.update_layout(
                        title=f'Детальный просмотр (окно {window_size} сек)',
                        xaxis_title='Отсчеты',
                        yaxis_title='Амплитуда',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_detailed, use_container_width=True)

            # Анализ стадий сна
            sleep_stages = analyzer.detect_sleep_stages(data.iloc[:, 0].values)
            
            # Вывод результатов
            st.subheader("Результаты анализа")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Глубокий сон", f"{sleep_stages['deep_sleep']:.1%}")
            with col2:
                st.metric("Промежуточная фаза", f"{sleep_stages['intermediate']:.1%}")
            with col3:
                st.metric("Бодрствование", f"{sleep_stages['wake']:.1%}")
                
            if show_statistics:
                st.subheader("Статистика сигнала")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write("Базовые характеристики:")
                    st.write(f"- Длительность записи: {len(data)/analyzer.sampling_rate:.1f} сек")
                    st.write(f"- Частота дискретизации: {analyzer.sampling_rate} Гц")
                    
                with stats_col2:
                    st.write("Спектральные характеристики:")
                    # Добавить спектральный анализ
                    pass
            
            # Экспорт результатов
            if st.button("Экспортировать результаты"):
                results = {
                    'sleep_stages': sleep_stages,
                    'analysis_params': {
                        'window_size': window_size,
                        'filter_type': filter_type
                    }
                }
                # Сохранение результатов в файл
                st.download_button(
                    "Скачать отчет",
                    data=pd.DataFrame(results).to_csv(),
                    file_name="sleep_analysis_report.csv",
                    mime="text/csv"
                )

def create_visualization(data: pd.DataFrame, predictions: np.ndarray, timestamps: np.ndarray, sampling_rate: int = 250):
    """Create visualization with signal and predicted states"""
    # Convert DataFrame to numpy array if needed
    signal_data = data['ECoG'].values if isinstance(data, pd.DataFrame) else data
    time = np.arange(len(signal_data)) / sampling_rate
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Исходный сигнал ЭКоГ", "Спектрограмма", "Состояния")
    )
    
    # Plot original signal
    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal_data,
            name="ЭКоГ",
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    try:
        # Add spectrogram with fixed parameters
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=sampling_rate,
            nperseg=1024,  # Fixed segment length
            noverlap=512,  # Fixed overlap
            detrend=False,
            scaling='density'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx + 1e-10),  # Added small constant to avoid log(0)
                x=t,
                y=f,
                colorscale='Viridis',
                name='Спектрограмма'
            ),
            row=2, col=1
        )
    except Exception as e:
        st.warning(f"Ошибка при создании спектрограммы: {str(e)}")
    
    # Plot predictions
    for state_id in np.unique(predictions):
        mask = predictions == state_id
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[mask],
                    y=[state_id] * np.sum(mask),
                    name=f"State {state_id}",
                    mode='markers',
                    marker=dict(
                        color=colors.get(state_id, "rgba(128, 128, 128, 0.3)"),
                        size=5
                    )
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title="Анализ ЭКоГ и состояний",
        xaxis_title="Время (с)",
        legend_title="Состояния"
    )
    
    return fig

if __name__ == "__main__":
    create_web_interface()