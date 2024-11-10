import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import mne
import os
import plotly.subplots as sp
from ml.catboost_inference import predict_states, get_state_name
from uuid import uuid4
import io

@st.cache_data(show_spinner=False)
def load_and_process_edf(file_buffer):
    temp_file = "temp.edf"
    with open(temp_file, "wb") as f:
        f.write(file_buffer.getbuffer())
    
    try:
        raw = mne.io.read_raw_edf(temp_file, preload=True)
        stat_data = raw.get_data()[0]
        sampling_rate = raw.info['sfreq']
        start_time = 0
        
        chunk_duration = 180
        chunk_samples = int(chunk_duration * sampling_rate)
        
        total_duration = len(raw.times) / sampling_rate
        st.info(f"""Загружен EDF файл:
            - Каналов: {len(raw.ch_names)}
            - Частота дискретизации: {sampling_rate} Гц
            - Общая длительность: {total_duration/60:.1f} минут
            - Текущий интервал: {start_time/60:.1f}-{(start_time + chunk_duration)/60:.1f} минут
        """)
        predictions, timestamps = predict_states(raw)

        return stat_data, sampling_rate, raw, start_time, total_duration, predictions, timestamps, chunk_samples, chunk_duration

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            

def create_visualization(data: np.ndarray, predictions: np.ndarray, timestamps: np.ndarray, sampling_rate: int = 250):
    """Create visualization with signal and predicted states"""
    time = np.arange(len(data)) / sampling_rate
    
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
        go.Scatter(y=data, name="ЭКоГ", line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Add spectrogram with higher resolution
    f, t, Sxx = signal.spectrogram(
        data,
        fs=sampling_rate,
        nperseg=1024,     # Размер окна
        noverlap=512,     # Перекрытие (должно быть < nperseg)
        detrend=False,
        scaling='density'
    )
    fig.add_trace(
        go.Heatmap(
            z=10 * np.log10(Sxx),
            x=t,
            y=f,
            colorscale='Viridis',
            name='Спектрограмма'
        ),
        row=2, col=1
    )
    
    # Colors for different states
    colors = {
        0: "rgba(251, 192, 147, 0.5)",  # Background
        1: "rgba(255, 0, 0, 0.5)",      # Spike-wave discharge
        2: "rgba(0, 0, 255, 0.5)",      # Delta sleep
        3: "rgba(0, 255, 0, 0.5)"       # Intermediate sleep
    }
    
    for state_id in np.unique(predictions):
        mask = predictions == state_id
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=timestamps[mask],
                    y=[state_id] * np.sum(mask),
                    name=get_state_name(state_id),
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
        legend_title="Состояия"
    )
    
    # Update y-axis for states plot
    fig.update_yaxes(
        title_text="Состояние",
        ticktext=[get_state_name(i) for i in range(4)],
        tickvals=list(range(4)),
        row=3, col=1
    )
    
    return fig

def save_predictions_to_edf(raw, predictions, timestamps):
    """Сохраняет предсказания как аннотации в EDF файле"""
    onset = []
    duration = []
    description = []
    
    state_names = {1: 'swd', 2: 'ds', 3: 'is'}
    
    current_state = predictions[0]
    start_time = timestamps[0]
    
    for i in range(1, len(predictions)):
        if predictions[i] != current_state or i == len(predictions) - 1:
            if current_state != 0:
                state_name = state_names[current_state]
                onset.append(float(start_time))
                duration.append(0.0)
                description.append(f"{state_name}1")
                onset.append(float(timestamps[i]))
                duration.append(0.0)
                description.append(f"{state_name}2")
            
            start_time = timestamps[i]
            current_state = predictions[i]
    
    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description
    )
    
    raw_predicted = raw.copy()
    raw_predicted.set_annotations(annotations)
    
    temp_file = f'{uuid4()}_predicted.edf'
    mne.export.export_raw(temp_file, raw_predicted, fmt='edf', overwrite=True)
    
    with open(temp_file, 'rb') as f:
        buffer = io.BytesIO(f.read())

    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return buffer

def create_web_interface():
    st.set_page_config(page_title="Анализ ЭКоГ сна крыс WAG/Rij", layout="wide")
    st.header("Анализ ЭКоГ сна крыс WAG/Rij")
    
    uploaded_file = st.file_uploader("Загрузите файл ЭКоГ (.csv, .edf)", type=['csv', 'edf'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file).iloc[:, 0].values
            else:
                stat_data, sampling_rate, raw, start_time, total_duration, predictions, timestamps, chunk_samples, chunk_duration = load_and_process_edf(uploaded_file)
                
                # Добавляем кнопку для скачивания файла с аннотациями
                if st.button("Скачать EDF файл с разметкой"):
                    buffer = save_predictions_to_edf(raw, predictions, timestamps)
                    st.download_button(
                        label="Скачать размеченный EDF файл",
                        data=buffer,
                        file_name=f"predicted_{uploaded_file.name}",
                        mime="application/octet-stream"
                    )

                total_minutes = int(total_duration / 60)
                periods = []
                for i in range(0, total_minutes, 3):
                    if i + 3 <= total_minutes:
                        periods.append(f"ЭКоГ. {i}-{i+3} минута(-ы)")
                    else:
                        periods.append(f"ЭКоГ. {i}-{total_minutes} минута(-ы)")
                
                selected_period = st.selectbox(
                    "Выберите период времени",
                    periods,
                    key='period_select'
                )
                
                start_minute = int(selected_period.split(' ')[1].split('-')[0])
                start_time = start_minute * 60
                
                start_sample = int(start_time * sampling_rate)
                data_chunk = raw.get_data(
                    start=start_sample,
                    stop=start_sample + chunk_samples
                )

                data = pd.DataFrame(data_chunk[0].T, columns=['ECoG'])
                
                chunk_time = np.arange(len(data)) / sampling_rate + start_time
                
                chunk_predictions = predictions[
                    (timestamps >= start_time) & 
                    (timestamps < start_time + chunk_duration)
                ]
                chunk_timestamps = timestamps[
                    (timestamps >= start_time) & 
                    (timestamps < start_time + chunk_duration)
                ]
                
                fig = sp.make_subplots(rows=1, cols=1, 
                                    shared_xaxes=True,
                                    vertical_spacing=0.02)
                
                fig.add_trace(
                    go.Scatter(
                        x=chunk_time,
                        y=data.iloc[:, 0],
                        name="Исходный сигнал",
                        line=dict(color='#1f77b4')
                    ),
                    row=1, col=1
                )
                            
                colors = {
                    0: "rgba(251, 192, 147, 0.5)",  # Background
                    1: "rgba(255, 0, 0, 0.5)",      # Spike-wave discharge
                    2: "rgba(0, 0, 255, 0.5)",      # Delta sleep
                    3: "rgba(0, 255, 0, 0.5)"       # Intermediate sleep
                }
    
                for state_id in np.unique(chunk_predictions):
                    mask = chunk_predictions == state_id
                    if np.any(mask):
                        changes = np.diff(mask.astype(int))
                        zone_starts = chunk_timestamps[:-1][changes == 1]
                        zone_ends = chunk_timestamps[:-1][changes == -1]
                        
                        if mask[0]:
                            zone_starts = np.insert(zone_starts, 0, chunk_timestamps[0])
                        if mask[-1]:
                            zone_ends = np.append(zone_ends, chunk_timestamps[-1])
                        
                        for start, end in zip(zone_starts, zone_ends):
                            fig.add_vrect(
                                x0=start,
                                x1=end,
                                fillcolor=colors.get(state_id, "rgba(128, 128, 128, 0.5)"),
                                layer="below",
                                line_width=0,
                                name=get_state_name(state_id),
                                legendgroup=f"state_{state_id}",
                                showlegend=bool(start == zone_starts[0])
                            )
                title = "Разметка " + selected_period
                fig.update_layout(
                    title=title,
                    height=600, 
                    showlegend=True,
                    xaxis_title="Время (с)"
                )
                
                st.plotly_chart(fig, use_container_width=True)

                fig_pie = go.Figure(data=[go.Pie(
                    labels=[get_state_name(i) for i in range(4)],
                    values=[np.sum(predictions == i) for i in range(4)],
                    hole=.3
                )])
                fig_pie.update_layout(title="Распределение состояний", width=500)
                st.plotly_chart(fig_pie)
                
        except Exception as e:
            st.error(f"Ошибка обработки данных: {str(e)}")

if __name__ == "__main__":
    create_web_interface()
