import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import joblib
import os
import plotly.express as px
from scipy import signal
import plotly.subplots as sp
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from functools import partial

# Загружаем переменные окружения
load_dotenv()

class SleepStageAnalyzer:
    """Класс для анализа стадий сна по ЭКоГ данным"""
    
    def __init__(self, model_path: str = os.getenv("MODEL_PATH", "sleep_stage_model.pth")):
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        self.max_workers = int(os.getenv("CPU_LIMIT", 1))
        
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

    @st.cache_data
    def process_chunk(self, chunk_data: np.ndarray) -> Dict:
        """Обработка отдельного чанка данных"""
        return self.detect_sleep_stages(chunk_data)
    
    def parallel_process(self, data: np.ndarray, chunk_size: int = 30000) -> List[Dict]:
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        
        return results

def create_web_interface():
    st.set_page_config(
        page_title=os.getenv("APP_TITLE", "Анализ ЭКоГ сна крыс WAG/Rij"),
        layout="wide"
    )
    
    # Боковая панель с настройками
    with st.sidebar:
        st.title("Настройки анализа")
        window_size = st.slider(
            "Размер окна анализа (сек)", 
            10, 60, 
            int(os.getenv("DEFAULT_WINDOW_SIZE", 30))
        )
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
            
            # Добавляем индикатор прогресса
            with st.spinner('Обработка данных...'):
                # Параллельная обработка данных
                chunk_results = analyzer.parallel_process(data.iloc[:, 0].values)
                
                # Агрегируем результаты
                sleep_stages = {
                    'deep_sleep': np.mean([r['deep_sleep'] for r in chunk_results]),
                    'intermediate': np.mean([r['intermediate'] for r in chunk_results]),
                    'wake': np.mean([r['wake'] for r in chunk_results])
                }
        
        # Визуализация данных
        st.subheader("Визуализация ЭКоГ данных")
        
        # Создаем вкладки для разных типов визуализации
        tab1, tab2, tab3 = st.tabs(["Временной анализ", "Частотный анализ", "Детальный просмотр"])
        
        with tab1:
            # Создаем график с двумя осями Y
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
                    title='Спектрограмма сигнала',
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
            
            # Вычисляем индексы для выбранного окна
            start_idx = window_start * analyzer.sampling_rate
            end_idx = start_idx + window_size * analyzer.sampling_rate
            
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

if __name__ == "__main__":
    create_web_interface()
