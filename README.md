# 🎵 Advanced Audio Edit Detector

**Detector avanzado de ediciones y manipulaciones en archivos de audio mediante análisis forense digital**


## 🔍 ¿Qué hace este programa?

Este software realiza un **análisis forense automatizado** de archivos de audio para detectar posibles manipulaciones, cortes y ediciones mediante técnicas avanzadas de procesamiento de señal digital.

### 🎯 Aplicaciones principales:
- **Análisis forense** de evidencias audio
- **Verificación de autenticidad** en procesos legales
- **Periodismo de investigación** y verificación de audios
- **Control de calidad** en producciones audio


## 🛠️ Características técnicas

### 🔬 Métodos de detección implementados:
- **Análisis multi-escala temporal** (20ms, 50ms, 100ms)
- **Detección espectral avanzada** (centroide, bandwidth, roll-off)
- **Cross-correlation** para identificar repeticiones
- **Clustering temporal** con DBSCAN
- **Umbrales adaptativos** basados en contexto local

### 📊 Salidas generadas:
- **Reportes técnicos** detallados en texto y HTML
- **Espectrogramas** interactivos con marcas de anomalías
- **Gráficos comparativos** de energía y características espectrales
- **Agrupamiento inteligente** de detecciones temporales


## 🚀 Instalación y uso rápido

```bash
# Instalar dependencias
pip install librosa numpy matplotlib scipy scikit-learn tqdm

# Uso básico
python detect_audio_edits_advanced.py audio.wav

# Análisis profesional con auto-calibración
python detect_audio_edits_advanced.py audio.wav --auto-calibrate --generate-html

# Máximo rigor (menos falsos positivos)
python detect_audio_edits_advanced.py audio.wav --sensitivity 8.0 --spectral 0.6

