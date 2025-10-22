# 🎵 Advanced Audio Edit Detector

**Detector avanzado de ediciones y manipulaciones en archivos de audio mediante análisis forense digital**

---

## 🔍 ¿Qué hace este programa?

Este software realiza un **análisis forense automatizado** de archivos de audio para detectar posibles manipulaciones, cortes y ediciones mediante técnicas avanzadas de procesamiento de señal digital.

### 🎯 Aplicaciones principales:
- **Análisis forense** de evidencias audio
- **Verificación de autenticidad** en procesos legales
- **Periodismo de investigación** y verificación de audios
- **Control de calidad** en producciones audio

---

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

---

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


Detector Avanzado de Ediciones en Audio: Análisis Forense Digital
🎯 ¿Qué hace este programa?

Este software realiza un análisis forense automatizado de archivos de audio para detectar posibles manipulaciones, ediciones o cortes. Utiliza técnicas avanzadas de procesamiento de señal digital para identificar discontinuidades que podrían indicar que el audio ha sido modificado después de su grabación original.
🔍 Problema que resuelve

En contextos forenses, periodísticos o de integridad de evidencias, es crucial determinar si un archivo de audio es original o ha sido editado. Las ediciones pueden incluir:

    Cortes y empalmes de diferentes segmentos

    Inserciones de audio externo

    Eliminaciones de partes del original

    Modificaciones que alteran el contenido

🛠️ Técnicas utilizadas y su fundamento científico
1. Análisis Multi-Escala Temporal
python

scales = [0.02, 0.05, 0.1]  # Diferentes resoluciones

¿Por qué? Las ediciones pueden ocurrir a diferentes escalas temporales:

    20ms: Detecta cortes muy precisos

    50ms: Balance ideal para la mayoría de ediciones

    100ms: Captura cambios más sutiles y graduales

2. Análisis Energético con Umbrales Adaptativos
python

energy_threshold = adaptive_threshold(energy_diff)

Técnica: Calcula la energía RMS por frame y aplica umbrales que se adaptan al contexto local.

Fundamento: Las ediciones suelen crear discontinuidades en la energía que son estadísticamente anómalas comparadas con las variaciones naturales del audio.
3. Distancia Coseno Espectral
python

spectral_similarity = 1 - cosine(spectrum1, spectrum2)

¿Por qué? Cada fuente de audio tiene una "huella digital espectral" única. Cuando se edita audio, se pueden mezclar diferentes:

    Micrófonos con respuestas frecuenciales distintas

    Ambientes con reverberaciones características

    Fuentes con diferentes características espectrales

4. Características Espectrales Avanzadas

    Centroide Espectral: Frecuencia promedio donde se concentra la energía

    Ancho de Banda: Distribución de energía en frecuencia

    Spectral Roll-off: Frecuencia donde se acumula el 85% de energía

    Flatness: Medida de sonido tonal vs. ruido

Aplicación: Cambios abruptos en estas características indican transiciones entre fuentes diferentes.
5. Clustering Temporal con DBSCAN
python

DBSCAN(eps=time_threshold, min_samples=1)

Objetivo: Agrupa detecciones cercanas en el tiempo para:

    Reducir falsos positivos

    Identificar regiones de interés

    Diferenciar entre eventos únicos y patrones repetitivos

6. Auto-Calibración Inteligente
python

def auto_calibrate(y, sr):

Innovación: El programa se adapta automáticamente a las características del audio:

    Audios cortos: Mayor sensibilidad

    Audios silenciosos: Ajuste de umbrales

    Audios largos: Optimización de recursos

📊 Metodología de Detección
Fase 1: Preprocesamiento

    Carga y normalización del audio

    División en frames solapados

    Cálculo de energía por segmento

Fase 2: Análisis Multi-Paramétrico

    Energía: Detecta cambios bruscos de amplitud

    Espectro: Identifica variaciones en contenido frecuencial

    Características avanzadas: Análisis de firmas acústicas

Fase 3: Fusión de Evidencias
python

combined_anomalies = energy_anomalies & spectral_anomalies

Estrategia: Solo se consideran ediciones cuando coinciden múltiples indicadores, reduciendo falsos positivos.
Fase 4: Post-Procesamiento

    Clustering de detecciones

    Validación temporal

    Generación de reportes

🎓 Base Científica
Teoría de Detección de Cambios

El programa implementa principios de Change Point Detection estadístico, buscando puntos donde las propiedades estadísticas de la señal cambian abruptamente.
Procesamiento de Señal Digital

    Transformada de Fourier para análisis espectral

    Ventanas de Hanning para reducir artefactos

    Suavizado adaptativo para preservar información

Machine Learning No Supervisado

    Clustering para agrupamiento temporal

    Detección de outliers para anomalías

🔬 Validación del Método
Ventajas sobre métodos simples:

    Robustez: No depende de un único parámetro

    Adaptabilidad: Funciona con diferentes calidades de audio

    Precisión: Combina múltiples evidencias

    Interpretabilidad: Proporciona justificación técnica

Limitaciones conocidas:

    Ediciones profesionales con crossfades pueden ser difíciles de detectar

    Cambios graduales naturales pueden generar falsos positivos

    Depende de la calidad de la grabación original

🏛️ Aplicaciones Prácticas
Forense Digital

    Análisis de evidencias audio

    Verificación de autenticidad en procesos legales

    Detección de manipulaciones en grabaciones

Periodismo de Investigación

    Verificación de audios filtrados

    Análisis de declaraciones públicas

    Detección de deepfakes audio

Control de Calidad

    Verificación de integridad en producciones

    Detección de errores en post-producción

    Análisis de consistencia en grabaciones

📈 Resultados e Interpretación
Salidas del programa:

    Reporte técnico con tiempos exactos de posibles ediciones

    Visualizaciones espectrales y temporales

    Agrupamiento inteligente de detecciones

    Recomendaciones para verificación manual

Interpretación de resultados:

    Alta confianza: Múltiples indicadores coincidentes

    Media confianza: Algunos indicadores presentes

    Baja confianza: Detecciones aisladas

🎯 Conclusión

Este programa representa un enfoque científico riguroso para la detección de ediciones en audio, combinando técnicas establecidas de procesamiento de señal con métodos modernos de machine learning. Proporciona a investigadores, periodistas y profesionales forenses una herramienta objetiva y reproducible para evaluar la integridad de grabaciones de audio.

⚠️ Nota importante: Los resultados deben ser interpretados por expertos y considerados como evidencia complementaria en un análisis forense completo.
