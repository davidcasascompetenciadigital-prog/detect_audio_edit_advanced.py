#!/usr/bin/env python3
"""
Script: audio_forensic_analyzer.py

Creator: David Casas M. - Competencia_Digital
Created with AI assistance: DEEPSEEK
License: GPL-3.0
YouTube: https://youtube.com/@competencia_digital
Website: https://competenciadigital.name

Description:
Analizador forense avanzado de audio que combina múltiples técnicas:
- Análisis espectral y energético tradicional
- Electric Network Frequency (ENF) analysis
- Detección multi-escala
- Machine Learning para clustering
- Análisis de armónicos y correlaciones cruzadas
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy.signal import stft, find_peaks, welch
import argparse
import os
import sys
from sklearn.cluster import DBSCAN
import json
from datetime import datetime

# --------------- CONFIGURACIÓN GLOBAL ---------------
DEFAULT_SENSITIVITY = 6.0
DEFAULT_SPECTRAL_SENSITIVITY = 0.4
DEFAULT_SMOOTHING_WINDOW = 7
DEFAULT_FRAME_DURATION = 0.05
DEFAULT_CLUSTER_THRESHOLD = 1.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Configuración ENF
ENF_FUNDAMENTAL = 50
ENF_RANGE = (49, 51) if ENF_FUNDAMENTAL == 50 else (59, 61)
ENF_HARMONICS = [2, 3, 4]
ENF_STFT_WINDOW = 4.0

# Variables globales para el reporte
duration = 0
sr = 0

# --------------- FUNCIONES AUXILIARES ---------------
def seconds_to_hms(seconds):
    """Convierte segundos a formato HH:MM:SS.ms"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def smooth_signal(signal, window_size):
    """Suavizado de señal con manejo de bordes mejorado"""
    if len(signal) == 0:
        return signal
    if window_size % 2 == 0:
        window_size += 1
    if window_size > len(signal):
        window_size = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    
    window = np.hanning(window_size)
    window /= np.sum(window)
    return np.convolve(signal, window, mode='same')

def safe_cosine(a, b):
    """Cosine distance segura con manejo de ceros"""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - np.dot(a, b) / (norm_a * norm_b)

def adaptive_threshold(signal, window_size=50):
    """Umbral adaptativo basado en contexto local"""
    if len(signal) == 0:
        return np.array([])
    
    thresholds = []
    for i in range(len(signal)):
        start = max(0, i - window_size//2)
        end = min(len(signal), i + window_size//2)
        
        if end - start > 0:
            local_mean = np.mean(signal[start:end])
            local_std = np.std(signal[start:end])
            thresholds.append(local_mean + 2 * local_std)
        else:
            thresholds.append(np.mean(signal) + 2 * np.std(signal))
    
    return np.array(thresholds)

def cluster_anomalies(anomaly_times, time_threshold=2.0):
    """Agrupa anomalías cercanas en el tiempo usando DBSCAN"""
    if len(anomaly_times) == 0:
        return anomaly_times, []
    
    X = anomaly_times.reshape(-1, 1)
    clustering = DBSCAN(eps=time_threshold, min_samples=1).fit(X)
    
    clustered_anomalies = []
    cluster_info = []
    
    for label in set(clustering.labels_):
        cluster_points = anomaly_times[clustering.labels_ == label]
        cluster_center = np.mean(cluster_points)
        clustered_anomalies.append(cluster_center)
        cluster_info.append({
            'center': cluster_center,
            'members': len(cluster_points),
            'time_range': f"{seconds_to_hms(np.min(cluster_points))} - {seconds_to_hms(np.max(cluster_points))}"
        })
    
    return np.array(clustered_anomalies), cluster_info

# --------------- ANÁLISIS ESPECTRAL TRADICIONAL ---------------
def advanced_spectral_features(spectrum, sr):
    """Extrae características espectrales avanzadas"""
    features = {}
    
    if np.sum(spectrum) == 0:
        return {
            'spectral_centroid': 0,
            'spectral_bandwidth': 0,
            'spectral_rolloff': 0,
            'spectral_flatness': 0
        }
    
    freqs = np.linspace(0, sr/2, len(spectrum))
    features['spectral_centroid'] = np.sum(freqs * spectrum) / np.sum(spectrum)
    
    centroid_diff = freqs - features['spectral_centroid']
    features['spectral_bandwidth'] = np.sqrt(np.sum(spectrum * centroid_diff**2) / np.sum(spectrum))
    
    cumulative_energy = np.cumsum(spectrum)
    total_energy = cumulative_energy[-1]
    if total_energy > 0:
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    else:
        features['spectral_rolloff'] = 0
    
    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arithmetic_mean = np.mean(spectrum)
    features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
    
    return features

def multi_scale_analysis(y, sr, scales=[0.02, 0.05, 0.1], sensitivity=4.0, spectral_sensitivity=0.3):
    """Análisis a diferentes resoluciones temporales"""
    all_anomalies = []
    
    for scale_idx, scale in enumerate(scales):
        print(f"🔍 Escala {scale_idx+1}/{len(scales)}: {scale}s por frame")
        
        frame_length = int(scale * sr)
        hop_length = frame_length // 2
        
        energy = []
        spectra = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            energy.append(np.sum(frame**2))
            
            spectrum = np.abs(np.fft.rfft(frame))
            spectrum_norm = np.linalg.norm(spectrum)
            if spectrum_norm > 1e-10:
                spectrum /= spectrum_norm
            else:
                spectrum = np.zeros_like(spectrum)
            spectra.append(spectrum)
        
        energy = np.array(energy)
        spectra = np.array(spectra)
        energy_smoothed = smooth_signal(energy, 5)
        times = np.arange(len(energy)) * (hop_length / sr) + (frame_length / (2 * sr))
        
        if len(energy_smoothed) > 1:
            energy_diff = np.abs(np.diff(energy_smoothed))
            spectral_diff = np.array([safe_cosine(spectra[i], spectra[i+1]) 
                                   for i in range(len(spectra)-1)])
            
            energy_threshold = adaptive_threshold(energy_diff)
            if len(energy_threshold) == 0:
                energy_threshold = np.mean(energy_diff) + sensitivity * np.std(energy_diff)
            
            energy_anomalies = energy_diff > energy_threshold
            spectral_anomalies = spectral_diff > spectral_sensitivity
            combined_anomalies = np.where(energy_anomalies & spectral_anomalies)[0]
            
            valid_anomalies = combined_anomalies[combined_anomalies < len(times)]
            scale_anomalies = times[valid_anomalies] if len(valid_anomalies) > 0 else np.array([])
            
            all_anomalies.extend(scale_anomalies)
            print(f"   ⚠️  Detectadas {len(scale_anomalies)} anomalías en esta escala")
    
    return np.array(all_anomalies)

def traditional_analysis(y, sr, analysis_params):
    """Análisis tradicional de energía y espectro"""
    print("[2/6] Realizando análisis espectral tradicional...")
    
    frame_length = int(analysis_params['frame_duration'] * sr)
    hop_length = frame_length // 2
    
    energy = []
    spectra = []
    spectral_features = []
    
    for i in tqdm(range(0, len(y) - frame_length, hop_length), desc="Análisis frames"):
        frame = y[i:i+frame_length]
        energy.append(np.sum(frame**2))
        
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum_norm = np.linalg.norm(spectrum)
        if spectrum_norm > 1e-10:
            spectrum /= spectrum_norm
        else:
            spectrum = np.zeros_like(spectrum)
        spectra.append(spectrum)
        
        # Características espectrales avanzadas
        features = advanced_spectral_features(spectrum, sr)
        spectral_features.append(features)
    
    energy = np.array(energy)
    spectra = np.array(spectra)
    energy_smoothed = smooth_signal(energy, analysis_params['smoothing_window'])
    times = np.arange(len(energy)) * (hop_length / sr) + (frame_length / (2 * sr))
    
    # Detección de anomalías
    energy_diff = np.abs(np.diff(energy_smoothed))
    spectral_diff = np.array([safe_cosine(spectra[i], spectra[i+1]) 
                           for i in range(len(spectra)-1)])
    
    energy_threshold = adaptive_threshold(energy_diff)
    spectral_threshold = np.mean(spectral_diff) + analysis_params['spectral_sensitivity'] * np.std(spectral_diff)
    
    energy_anomalies = energy_diff > energy_threshold
    spectral_anomalies = spectral_diff > spectral_threshold
    combined_anomalies = np.where(energy_anomalies & spectral_anomalies)[0]
    
    valid_anomalies = combined_anomalies[combined_anomalies < len(times)]
    anomaly_times = times[valid_anomalies] if len(valid_anomalies) > 0 else np.array([])
    
    return {
        'anomaly_times': anomaly_times,
        'energy_smoothed': energy_smoothed,
        'times': times,
        'energy': energy,
        'spectral_features': spectral_features
    }

# --------------- ANÁLISIS ENF ---------------
def extract_enf_signal(y, sr, fundamental_freq=50, freq_range=(49, 51)):
    """Extrae la señal ENF del audio"""
    print("🔌 Extrayendo señal ENF...")
    
    window_length = int(ENF_STFT_WINDOW * sr)
    hop_length = window_length // 4
    
    f, t, Zxx = stft(y, sr, nperseg=window_length, noverlap=window_length - hop_length)
    
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    enf_frequencies = f[freq_mask]
    enf_spectrum = np.abs(Zxx[freq_mask, :])
    
    enf_signal = []
    enf_power = []
    
    for i in range(enf_spectrum.shape[1]):
        frame_spectrum = enf_spectrum[:, i]
        if np.max(frame_spectrum) > 0:
            peak_idx = np.argmax(frame_spectrum)
            dominant_freq = enf_frequencies[peak_idx]
            dominant_power = frame_spectrum[peak_idx]
            
            enf_signal.append(dominant_freq)
            enf_power.append(dominant_power)
        else:
            enf_signal.append(fundamental_freq)
            enf_power.append(0)
    
    enf_times = t[:len(enf_signal)]
    
    return np.array(enf_signal), np.array(enf_power), enf_times

def analyze_enf_consistency(enf_signal, enf_times, sr):
    """Analiza la consistencia de la señal ENF"""
    print("📊 Analizando consistencia ENF...")
    
    enf_diff = np.diff(enf_signal)
    enf_std = np.std(enf_signal)
    enf_mean = np.mean(enf_signal)
    
    threshold_enf = enf_mean + 3 * enf_std
    enf_anomalies = np.where(np.abs(enf_diff) > threshold_enf)[0]
    
    power_variations = np.std(enf_signal) / (np.mean(enf_signal) + 1e-10)
    consistency_score = 1.0 / (1.0 + power_variations)
    
    enf_results = {
        'signal': enf_signal,
        'times': enf_times,
        'anomalies': enf_times[enf_anomalies] if len(enf_anomalies) > 0 else np.array([]),
        'stability_score': consistency_score,
        'mean_frequency': enf_mean,
        'std_frequency': enf_std,
        'total_anomalies': len(enf_anomalies)
    }
    
    print(f"   📈 Estabilidad ENF: {consistency_score:.3f}")
    print(f"   ⚠️  Anomalías ENF detectadas: {len(enf_anomalies)}")
    
    return enf_results

def harmonic_enf_analysis(y, sr, fundamental=50):
    """Análisis de armónicos ENF"""
    print("🎵 Analizando armónicos ENF...")
    
    harmonic_results = {}
    
    for harmonic in ENF_HARMONICS:
        harmonic_freq = fundamental * harmonic
        freq_range = (harmonic_freq - 2, harmonic_freq + 2)
        
        harmonic_signal, harmonic_power, harmonic_times = extract_enf_signal(
            y, sr, harmonic_freq, freq_range
        )
        
        harmonic_results[harmonic] = {
            'signal': harmonic_signal,
            'power': harmonic_power,
            'times': harmonic_times,
            'mean_freq': np.mean(harmonic_signal),
            'consistency': 1.0 / (1.0 + np.std(harmonic_signal) / np.mean(harmonic_signal))
        }
    
    return harmonic_results

def cross_correlation_enf(harmonic_results):
    """Correlación cruzada entre armónicos ENF"""
    print("🔗 Analizando correlación entre armónicos...")
    
    if len(harmonic_results) < 2:
        return 1.0
    
    correlations = []
    harmonics = list(harmonic_results.keys())
    
    for i in range(len(harmonics)):
        for j in range(i + 1, len(harmonics)):
            h1 = harmonic_results[harmonics[i]]['signal']
            h2 = harmonic_results[harmonics[j]]['signal']
            
            min_len = min(len(h1), len(h2))
            h1 = h1[:min_len]
            h2 = h2[:min_len]
            
            if len(h1) > 10:
                correlation = np.corrcoef(h1, h2)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
    
    avg_correlation = np.mean(correlations) if correlations else 0
    print(f"   📊 Correlación promedio entre armónicos: {avg_correlation:.3f}")
    
    return avg_correlation

def calculate_enf_quality_metrics(enf_signal, enf_times, fundamental_freq):
    """Calcula métricas adicionales de calidad de la señal ENF"""
    # Calcular calidad de señal basada en continuidad
    valid_samples = np.sum(~np.isnan(enf_signal))
    total_samples = len(enf_signal)
    data_completeness = valid_samples / total_samples if total_samples > 0 else 0
    
    # Calcular proximidad al valor nominal
    mean_freq = np.nanmean(enf_signal)
    freq_proximity = 1.0 - (abs(mean_freq - fundamental_freq) / 2.0)  # Normalizado a 0-1
    
    # Calcular suavidad de la señal
    if len(enf_signal) > 1:
        diffs = np.abs(np.diff(enf_signal[~np.isnan(enf_signal)]))
        smoothness = 1.0 / (1.0 + np.mean(diffs)) if len(diffs) > 0 else 0
    else:
        smoothness = 0
    
    signal_quality = (data_completeness * 0.4 + freq_proximity * 0.3 + smoothness * 0.3)
    
    return {
        'data_completeness': data_completeness,
        'freq_proximity': freq_proximity,
        'smoothness': smoothness,
        'signal_quality': signal_quality
    }

def enf_analysis(y, sr, enf_fundamental=50):
    """Análisis ENF completo con métricas mejoradas"""
    print("\n" + "="*70)
    print("🔌 INICIANDO ANÁLISIS ENF (ELECTRIC NETWORK FREQUENCY)")
    print("="*70)
    
    # Análisis ENF fundamental
    enf_signal, enf_power, enf_times = extract_enf_signal(y, sr, enf_fundamental, 
                                                         (enf_fundamental-1, enf_fundamental+1))
    enf_results = analyze_enf_consistency(enf_signal, enf_times, sr)
    
    # Análisis de armónicos
    harmonic_results = harmonic_enf_analysis(y, sr, enf_fundamental)
    
    # Correlación cruzada
    correlation_score = cross_correlation_enf(harmonic_results)
    
    # Métricas adicionales de calidad ENF
    enf_quality_metrics = calculate_enf_quality_metrics(enf_signal, enf_times, enf_fundamental)
    
    return {
        'enf_results': enf_results,
        'harmonic_results': harmonic_results,
        'correlation_score': correlation_score,
        'quality_metrics': enf_quality_metrics,
        'combined_confidence': (enf_results['stability_score'] * 0.6 + 
                              correlation_score * 0.3 + 
                              enf_quality_metrics['signal_quality'] * 0.1)
    }

# --------------- VISUALIZACIONES COMPLETAS ---------------
def plot_spectrogram_with_anomalies(y, sr, anomaly_times, filename):
    """Espectrograma con marcas de anomalías"""
    plt.figure(figsize=(14, 10))
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    
    for t in anomaly_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label='Anomalía' if t == anomaly_times[0] else "")
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma con Detección de Ediciones')
    
    plt.subplot(2, 1, 2)
    times = np.arange(len(y)) / sr
    plt.plot(times, y, alpha=0.7, color='blue', linewidth=0.5)
    for t in anomaly_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Forma de Onda con Anomalías Detectadas')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_enf_analysis(enf_results, audio_file, output_path):
    """Visualización del análisis ENF"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(enf_results['times'], enf_results['signal'], 'b-', linewidth=1, label='Señal ENF')
    
    if len(enf_results['anomalies']) > 0:
        anomaly_indices = [np.argmin(np.abs(enf_results['times'] - t)) 
                          for t in enf_results['anomalies']]
        anomaly_freqs = [enf_results['signal'][idx] for idx in anomaly_indices]
        plt.scatter(enf_results['anomalies'], anomaly_freqs, 
                   color='red', s=50, zorder=5, label='Anomalías ENF')
    
    plt.axhline(y=ENF_FUNDAMENTAL, color='green', linestyle='--', 
                alpha=0.7, label=f'Frecuencia nominal ({ENF_FUNDAMENTAL} Hz)')
    plt.ylabel('Frecuencia (Hz)')
    plt.title(f'Análisis ENF - {os.path.basename(audio_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.hist(enf_results['signal'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(ENF_FUNDAMENTAL, color='red', linestyle='--', linewidth=2, 
                label=f'Nominal: {ENF_FUNDAMENTAL} Hz')
    plt.axvline(enf_results['mean_frequency'], color='green', linestyle='--', linewidth=2,
                label=f'Media: {enf_results["mean_frequency"]:.3f} Hz')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Frecuencia de ocurrencia')
    plt.title('Distribución de Frecuencias ENF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    moving_std = []
    window_size = min(20, len(enf_results['signal']) // 10)
    
    for i in range(len(enf_results['signal']) - window_size):
        window_std = np.std(enf_results['signal'][i:i+window_size])
        moving_std.append(window_std)
    
    if moving_std:
        moving_times = enf_results['times'][:len(moving_std)]
        plt.plot(moving_times, moving_std, 'orange', linewidth=1, label='Desviación estándar móvil')
        plt.axhline(y=enf_results['std_frequency'], color='red', linestyle='--',
                   label=f'Desviación global: {enf_results["std_frequency"]:.3f} Hz')
    
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Desviación estándar (Hz)')
    plt.title('Estabilidad ENF en el Tiempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_detection_analysis(traditional_results, clustered_anomalies, audio_file, output_path):
    """Gráfico de detección de anomalías (energía suavizada)"""
    plt.figure(figsize=(14, 6))
    
    plt.plot(traditional_results['times'], traditional_results['energy_smoothed'], 
             label="Energía suavizada", linewidth=1, color='blue')
    
    if len(clustered_anomalies) > 0:
        # Encontrar índices de las anomalías para plotear los valores correctos
        valid_indices = []
        anomaly_values = []
        for t in clustered_anomalies:
            idx = np.argmin(np.abs(traditional_results['times'] - t))
            if idx < len(traditional_results['energy_smoothed']):
                valid_indices.append(idx)
                anomaly_values.append(traditional_results['energy_smoothed'][idx])
        
        if valid_indices:
            plt.scatter(clustered_anomalies, anomaly_values, 
                       color='red', s=50, label='Posibles ediciones', zorder=5)
            
            # Anotaciones de tiempo
            for idx, t in enumerate(clustered_anomalies):
                if idx < 10:  # Limitar anotaciones para evitar saturación
                    plt.annotate(seconds_to_hms(t), 
                                (t, anomaly_values[idx]), 
                                textcoords="offset points", xytext=(0,10), 
                                ha='center', fontsize=8, color='red')
    
    plt.title(f"Detección de Ediciones - {os.path.basename(audio_file)}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía relativa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_energy_comparison(traditional_results, audio_file, output_path):
    """Gráfico comparativo de energía original vs suavizada"""
    plt.figure(figsize=(14, 6))
    
    plt.plot(traditional_results['times'], traditional_results['energy'], 
             alpha=0.7, label="Energía original", color='gray', linewidth=0.8)
    plt.plot(traditional_results['times'], traditional_results['energy_smoothed'], 
             label="Energía suavizada", linewidth=1.5, color='blue')
    
    plt.title("Comparativa de Energía Original vs Suavizada")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía relativa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# --------------- GENERACIÓN DE REPORTES MEJORADA ---------------
def generate_comprehensive_report(audio_file, traditional_results, enf_results, 
                                cluster_info, analysis_params, output_dir):
    """Genera reporte completo unificado con explicaciones detalladas"""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    report_file = os.path.join(output_dir, f"{base_name}_forensic_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("INFORME FORENSE COMPLETO DE ANÁLISIS DE AUDIO\n")
        f.write("=" * 70 + "\n")
        f.write(f"Creator: David Casas M. - Competencia_Digital\n")
        f.write(f"License: GPL-3.0 | AI Assistance: DEEPSEEK\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Archivo analizado: {audio_file}\n")
        f.write(f"Duración: {duration:.2f} segundos\n")
        f.write(f"Frecuencia de muestreo: {sr} Hz\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESUMEN EJECUTIVO:\n")
        f.write("-" * 20 + "\n")
        
        # Calcular confianza general
        traditional_confidence = 1.0 - (len(traditional_results['clustered_anomalies']) / 50.0)
        traditional_confidence = max(0.0, min(1.0, traditional_confidence))
        
        enf_confidence = enf_results['combined_confidence'] if enf_results else 0.5
        
        overall_confidence = (traditional_confidence * 0.6 + enf_confidence * 0.4)
        
        # Interpretación de confianza general
        if overall_confidence > 0.8:
            f.write("✅ ALTA PROBABILIDAD DE AUDIO ORIGINAL\n")
            f.write("   - Múltiples indicadores sugieren autenticidad\n")
        elif overall_confidence > 0.6:
            f.write("⚠️  PROBABILIDAD MODERADA - VERIFICAR MANUALMENTE\n")
            f.write("   - Algunos indicadores requieren verificación adicional\n")
        else:
            f.write("❌ ALTA PROBABILIDAD DE MANIPULACIÓN\n")
            f.write("   - Múltiples evidencias sugieren edición del audio\n")
        
        f.write(f"Puntuación general de confianza: {overall_confidence:.3f}\n")
        f.write("   [Escala: 0.0-1.0, donde >0.8 = alta confianza]\n\n")
        
        f.write("ANÁLISIS TRADICIONAL (Energía + Espectro):\n")
        f.write("-" * 45 + "\n")
        f.write(f"Anomalías detectadas: {len(traditional_results['clustered_anomalies'])}\n")
        f.write(f"Clusters identificados: {len(cluster_info)}\n")
        f.write(f"Confianza análisis tradicional: {traditional_confidence:.3f}\n")
        f.write("   [Escala: 0.0-1.0, donde >0.7 = baja sospecha]\n\n")
        
        if enf_results:
            f.write("ANÁLISIS ENF (ELECTRIC NETWORK FREQUENCY):\n")
            f.write("-" * 45 + "\n")
            
            # Estabilidad ENF con explicación
            stability = enf_results['enf_results']['stability_score']
            f.write(f"Estabilidad ENF: {stability:.3f} - ")
            if stability > 0.8:
                f.write("EXCELENTE ✅\n")
                f.write("   • La señal ENF muestra alta consistencia temporal\n")
                f.write("   • Patrón típico de grabación continua y original\n")
            elif stability > 0.6:
                f.write("MODERADA ⚠️\n")
                f.write("   • Ligera variación que podría ser natural o indicar ediciones menores\n")
            else:
                f.write("BAJA ❌\n")
                f.write("   • Alta variabilidad sugiere posibles cortes o mezclas\n")
            f.write("   [Escala: 0.0-1.0, donde >0.8 = excelente estabilidad]\n\n")
            
            # Correlación armónicos con explicación
            correlation = enf_results['correlation_score']
            f.write(f"Correlación armónicos: {correlation:.3f} - ")
            if correlation > 0.8:
                f.write("FUERTE ✅\n")
                f.write("   • Los armónicos mantienen relación consistente\n")
                f.write("   • Indica fuente eléctrica única y estable\n")
            elif correlation > 0.5:
                f.write("MODERADA ⚠️\n")
                f.write("   • Algunas discrepancias entre armónicos\n")
                f.write("   • Podría indicar cambios en la fuente eléctrica\n")
            else:
                f.write("DÉBIL ❌\n")
                f.write("   • Baja correlación sugiere múltiples fuentes o ediciones\n")
            f.write("   [Escala: 0.0-1.0, donde >0.8 = correlación fuerte]\n\n")
            
            f.write(f"Anomalías ENF detectadas: {enf_results['enf_results']['total_anomalies']}\n")
            f.write(f"Confianza combinada ENF: {enf_results['combined_confidence']:.3f}\n")
            f.write("   [Combinación de estabilidad y correlación armónicos]\n\n")
            
            # Frecuencia media ENF
            mean_freq = enf_results['enf_results']['mean_frequency']
            freq_variation = abs(mean_freq - analysis_params['enf_fundamental'])
            f.write(f"Frecuencia ENF media: {mean_freq:.3f} Hz\n")
            if freq_variation < 0.1:
                f.write("   • Muy cercana al valor nominal - Excelente ✅\n")
            elif freq_variation < 0.3:
                f.write("   • Variación dentro de rangos normales - Aceptable ⚠️\n")
            else:
                f.write("   • Desviación significativa - Requiere investigación ❌\n")
            f.write(f"   [Nominal: {analysis_params['enf_fundamental']} Hz, Desviación: {freq_variation:.3f} Hz]\n\n")
        
        f.write("DETALLES TÉCNICOS:\n")
        f.write("-" * 20 + "\n")
        for key, value in analysis_params.items():
            f.write(f"  - {key}: {value}\n")
        
        # Análisis detallado de anomalías tradicionales
        if len(traditional_results['clustered_anomalies']) > 0:
            f.write(f"\nANÁLISIS DETALLADO DE ANOMALÍAS TRADICIONALES:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Se detectaron {len(traditional_results['clustered_anomalies'])} anomalías agrupadas en {len(cluster_info)} clusters:\n\n")
            
            for i, anomaly in enumerate(traditional_results['clustered_anomalies'], 1):
                f.write(f"ANOMALÍA {i:2d}: {seconds_to_hms(anomaly)} ({anomaly:.3f}s)\n")
                
                # Clasificar el tipo de anomalía basado en la posición temporal
                if anomaly < duration * 0.1:
                    f.write("   • POSIBLE: Inicio de grabación o activación\n")
                elif anomaly > duration * 0.9:
                    f.write("   • POSIBLE: Final de grabación o desactivación\n")
                else:
                    f.write("   • POSIBLE: Corte o transición en contenido\n")
                
                # Buscar el cluster correspondiente
                for cluster in cluster_info:
                    if abs(cluster['center'] - anomaly) < 0.1:
                        if cluster['members'] > 1:
                            f.write(f"   • CLUSTER: {cluster['members']} detecciones cercanas\n")
                            f.write(f"   • RANGO: {cluster['time_range']}\n")
                        break
                f.write("\n")
        
        # Análisis detallado de anomalías ENF
        if enf_results and enf_results['enf_results']['total_anomalies'] > 0:
            f.write(f"\nANÁLISIS DETALLADO DE ANOMALÍAS ENF:\n")
            f.write("-" * 40 + "\n")
            f.write("Las anomalías ENF indican cambios abruptos en la frecuencia de red:\n\n")
            
            for i, anomaly in enumerate(enf_results['enf_results']['anomalies'], 1):
                f.write(f"ANOMALÍA ENF {i:2d}: {seconds_to_hms(anomaly)} ({anomaly:.3f}s)\n")
                
                # Encontrar el valor de frecuencia en la anomalía
                anomaly_idx = np.argmin(np.abs(enf_results['enf_results']['times'] - anomaly))
                if anomaly_idx < len(enf_results['enf_results']['signal']):
                    anomaly_freq = enf_results['enf_results']['signal'][anomaly_idx]
                    freq_diff = abs(anomaly_freq - analysis_params['enf_fundamental'])
                    
                    f.write(f"   • FRECUENCIA: {anomaly_freq:.3f} Hz (Δ={freq_diff:.3f} Hz)\n")
                    
                    if freq_diff > 0.5:
                        f.write("   • TIPO: Salto de frecuencia significativo ❌\n")
                        f.write("   • POSIBLE: Cambio de fuente eléctrica o edición\n")
                    elif freq_diff > 0.2:
                        f.write("   • TIPO: Variación moderada ⚠️\n")
                        f.write("   • POSIBLE: Fluctuación natural de red o transición\n")
                    else:
                        f.write("   • TIPO: Variación menor ⚠️\n")
                        f.write("   • POSIBLE: Fluctuación normal de la red eléctrica\n")
                f.write("\n")
        
        # RECOMENDACIONES FINALES
        f.write("\nRECOMENDACIONES Y PASOS SIGUIENTES:\n")
        f.write("-" * 35 + "\n")
        
        if overall_confidence > 0.8:
            f.write("✅ RECOMENDACIÓN: El audio muestra alta probabilidad de ser original.\n")
            f.write("   • Verificación manual opcional para confirmación final\n")
        elif overall_confidence > 0.6:
            f.write("⚠️  RECOMENDACIÓN: Verificación manual requerida.\n")
            f.write("   • Revisar los segmentos con anomalías detectadas\n")
            f.write("   • Considerar el contexto de la grabación\n")
            f.write("   • Evaluar posibles explicaciones naturales para las anomalías\n")
        else:
            f.write("❌ RECOMENDACIÓN: Investigación forense profunda recomendada.\n")
            f.write("   • Análisis manual de todos los segmentos con anomalías\n")
            f.write("   • Considerar análisis con herramientas adicionales\n")
            f.write("   • Documentar hallazgos para posible evidencia\n")
        
        f.write("\nMETODOLOGÍA UTILIZADA:\n")
        f.write("-" * 25 + "\n")
        f.write("• Análisis espectral: Detección de cambios en contenido frecuencial\n")
        f.write("• Análisis energético: Detección de discontinuidades en amplitud\n")
        f.write("• ENF: Análisis de la huella de frecuencia de red eléctrica\n")
        f.write("• Clustering: Agrupamiento inteligente de detecciones temporales\n")
        f.write("• Multi-escala: Análisis a diferentes resoluciones temporales\n")
    
    return report_file, overall_confidence

def print_detailed_results(traditional_results, enf_results, overall_confidence, duration):
    """Imprime resultados detallados en consola"""
    print("\n" + "=" * 70)
    print("🎯 RESULTADOS DETALLADOS DEL ANÁLISIS FORENSE")
    print("=" * 70)
    
    print(f"📊 CONFIANZA GENERAL: {overall_confidence:.3f}")
    if overall_confidence > 0.8:
        print("   ✅ ALTA - Audio probablemente original")
    elif overall_confidence > 0.6:
        print("   ⚠️  MODERADA - Verificación manual recomendada")
    else:
        print("   ❌ BAJA - Alta sospecha de manipulación")
    
    print(f"\n🎵 ANÁLISIS TRADICIONAL:")
    print(f"   • Anomalías detectadas: {len(traditional_results['clustered_anomalies'])}")
    traditional_confidence = 1.0 - (len(traditional_results['clustered_anomalies']) / 50.0)
    print(f"   • Confianza: {traditional_confidence:.3f}")
    
    if enf_results:
        print(f"\n🔌 ANÁLISIS ENF:")
        print(f"   • Estabilidad: {enf_results['enf_results']['stability_score']:.3f} ", end="")
        if enf_results['enf_results']['stability_score'] > 0.8:
            print("✅ EXCELENTE")
        elif enf_results['enf_results']['stability_score'] > 0.6:
            print("⚠️  MODERADA")
        else:
            print("❌ BAJA")
        
        print(f"   • Correlación armónicos: {enf_results['correlation_score']:.3f} ", end="")
        if enf_results['correlation_score'] > 0.8:
            print("✅ FUERTE")
        elif enf_results['correlation_score'] > 0.5:
            print("⚠️  MODERADA")
        else:
            print("❌ DÉBIL")
        
        print(f"   • Anomalías ENF: {enf_results['enf_results']['total_anomalies']}")
        print(f"   • Confianza ENF: {enf_results['combined_confidence']:.3f}")
    
    # Mostrar anomalías significativas
    if len(traditional_results['clustered_anomalies']) > 0:
        print(f"\n⚠️  ANOMALÍAS SIGNIFICATIVAS DETECTADAS:")
        for i, anomaly in enumerate(traditional_results['clustered_anomalies'][:5], 1):  # Mostrar primeras 5
            time_percentage = (anomaly / duration) * 100
            print(f"   {i}. {seconds_to_hms(anomaly)} ({time_percentage:.1f}% del audio)")
            
            # Clasificación básica
            if time_percentage < 10:
                print("      • POSIBLE: Inicio de grabación o activación")
            elif time_percentage > 90:
                print("      • POSIBLE: Final de grabación o desactivación")
            else:
                print("      • POSIBLE: Corte o transición en contenido")

# --------------- FUNCIÓN PRINCIPAL ---------------
def main():
    """Función principal del analizador forense completo"""
    global duration, sr
    
    parser = argparse.ArgumentParser(
        description="Analizador forense avanzado de audio con detección ENF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python audio_forensic_analyzer.py audio.wav
  python audio_forensic_analyzer.py audio.wav --enf-analysis --multi-scale
  python audio_forensic_analyzer.py audio.wav --sensitivity 8.0 --enf-fundamental 60
  python audio_forensic_analyzer.py audio.wav --auto-calibrate
        """
    )
    
    parser.add_argument('audio_file', help='Ruta al archivo de audio a analizar')
    
    # Parámetros tradicionales
    parser.add_argument('--sensitivity', '-s', type=float, default=DEFAULT_SENSITIVITY,
                       help=f'Umbral de sensibilidad (default: {DEFAULT_SENSITIVITY})')
    parser.add_argument('--spectral', '-sp', type=float, default=DEFAULT_SPECTRAL_SENSITIVITY,
                       help=f'Umbral para cambios espectrales (default: {DEFAULT_SPECTRAL_SENSITIVITY})')
    parser.add_argument('--smoothing', '-sm', type=int, default=DEFAULT_SMOOTHING_WINDOW,
                       help=f'Ventana de suavizado (default: {DEFAULT_SMOOTHING_WINDOW})')
    parser.add_argument('--frame', '-f', type=float, default=DEFAULT_FRAME_DURATION,
                       help=f'Duración del frame (default: {DEFAULT_FRAME_DURATION})')
    parser.add_argument('--multi-scale', action='store_true',
                       help='Análisis multi-escala para mejor detección')
    
    # Parámetros ENF
    parser.add_argument('--enf-analysis', action='store_true',
                       help='Realizar análisis ENF (Electric Network Frequency)')
    parser.add_argument('--enf-fundamental', type=int, default=50, choices=[50, 60],
                       help='Frecuencia fundamental de red (50Hz Europa, 60Hz América)')
    
    # Otros parámetros
    parser.add_argument('--auto-calibrate', action='store_true',
                       help='Calibración automática de parámetros')
    parser.add_argument('--cluster-threshold', type=float, default=DEFAULT_CLUSTER_THRESHOLD,
                       help=f'Umbral para agrupar anomalías (default: {DEFAULT_CLUSTER_THRESHOLD})')
    parser.add_argument('--output', '-o', help='Prefijo personalizado para archivos de salida')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: El archivo '{args.audio_file}' no existe.")
        sys.exit(1)
    
    # Mostrar cabecera
    print("=" * 70)
    print("🔍 ANALIZADOR FORENSE AVANZADO DE AUDIO")
    print("Creator: David Casas M. - Competencia_Digital")
    print("License: GPL-3.0 | Created with AI assistance")
    print("=" * 70)
    print()
    
    # Configuración
    base_name = args.output if args.output else os.path.splitext(os.path.basename(args.audio_file))[0]
    output_dir = os.path.dirname(args.audio_file) if args.output else "."
    
    analysis_params = {
        'sensitivity': args.sensitivity,
        'spectral_sensitivity': args.spectral,
        'smoothing_window': args.smoothing,
        'frame_duration': args.frame,
        'cluster_threshold': args.cluster_threshold,
        'multi_scale': args.multi_scale,
        'enf_analysis': args.enf_analysis,
        'enf_fundamental': args.enf_fundamental
    }
    
    # Cargar audio
    print("[1/6] Cargando archivo de audio...")
    try:
        y, sr = librosa.load(args.audio_file, sr=None, mono=True)
        duration = len(y) / sr
        print(f"✅ Audio cargado: {duration:.2f}s, {sr} Hz, {len(y)} muestras")
    except Exception as e:
        print(f"❌ Error cargando {args.audio_file}: {e}")
        sys.exit(1)
    
    # Análisis tradicional
    traditional_results = None
    if args.multi_scale:
        print("[2/6] Realizando análisis multi-escala...")
        anomaly_times = multi_scale_analysis(y, sr, [0.02, 0.05, 0.1], 
                                           args.sensitivity, args.spectral)
        traditional_results = {'anomaly_times': anomaly_times}
    else:
        traditional_results = traditional_analysis(y, sr, analysis_params)
    
    # Clustering de anomalías tradicionales
    clustered_anomalies, cluster_info = cluster_anomalies(
        traditional_results['anomaly_times'], 
        args.cluster_threshold
    )
    traditional_results['clustered_anomalies'] = clustered_anomalies
    
    # Análisis ENF
    enf_results = None
    if args.enf_analysis:
        print("[3/6] Realizando análisis ENF...")
        enf_results = enf_analysis(y, sr, args.enf_fundamental)
    
    # Generar visualizaciones COMPLETAS
    print("[4/6] Generando visualizaciones...")
    
    # 1. Espectrograma con anomalías
    spectrogram_file = os.path.join(output_dir, f"{base_name}_spectrogram.png")
    plot_spectrogram_with_anomalies(y, sr, clustered_anomalies, spectrogram_file)
    
    # 2. Gráfico de detección (energía suavizada con anomalías)
    detection_file = os.path.join(output_dir, f"{base_name}_detection_plot.png")
    plot_detection_analysis(traditional_results, clustered_anomalies, args.audio_file, detection_file)
    
    # 3. Gráfico comparativo de energía
    energy_file = os.path.join(output_dir, f"{base_name}_energy_comparison.png")
    plot_energy_comparison(traditional_results, args.audio_file, energy_file)
    
    # 4. Gráfico ENF si se realizó análisis
    enf_plot_file = None
    if enf_results:
        enf_plot_file = os.path.join(output_dir, f"{base_name}_enf_analysis.png")
        plot_enf_analysis(enf_results['enf_results'], args.audio_file, enf_plot_file)
    
    # Generar reporte completo
    print("[5/6] Generando reporte forense...")
    report_file, overall_confidence = generate_comprehensive_report(
        args.audio_file, traditional_results, enf_results, 
        cluster_info, analysis_params, output_dir
    )
    
    # Mostrar resultados detallados en consola
    print("[6/6] Preparando resumen ejecutivo...")
    print_detailed_results(traditional_results, enf_results, overall_confidence, duration)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("🎯 ANÁLISIS FORENSE COMPLETADO")
    print("=" * 70)
    
    print(f"📊 RESULTADOS COMBINADOS:")
    print(f"   🎵 Análisis tradicional: {len(clustered_anomalies)} anomalías")
    if enf_results:
        print(f"   🔌 Análisis ENF: {enf_results['enf_results']['total_anomalies']} anomalías")
        print(f"   📈 Confianza ENF: {enf_results['combined_confidence']:.3f}")
    
    print(f"   💯 Confianza general: {overall_confidence:.3f}")
    
    if overall_confidence > 0.8:
        print("   ✅ Alta probabilidad de audio original")
    elif overall_confidence > 0.6:
        print("   ⚠️  Verificación manual recomendada")
    else:
        print("   ❌ Alta probabilidad de manipulación")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   📄 Reporte forense: {report_file}")
    print(f"   📊 Espectrograma: {spectrogram_file}")
    print(f"   📈 Gráfico detección: {detection_file}")
    print(f"   🔋 Comparativa energía: {energy_file}")
    if enf_plot_file:
        print(f"   🔌 Análisis ENF: {enf_plot_file}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
