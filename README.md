Detect Audio Edition Forensic Analyzer

![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)
![Version](https://img.shields.io/badge/Version-1.0-green.svg)

Analizador de ediciones en archivos de audio WAV

Descripción

Analizador forense avanzado de audio que combina múltiples técnicas de detección de manipulaciones y ediciones. Esta herramienta está diseñada para asistir en investigaciones forenses, verificaciones periodísticas y análisis de integridad de evidencias audio.

El software implementa un enfoque multi-metodológico que incluye análisis espectral tradicional, detección de Electric Network Frequency (ENF), clustering inteligente de anomalías y análisis multi-escala temporal.
Características Técnicas
Métodos de Análisis Implementados

    Análisis Espectro-Energético Tradicional: Detección de discontinuidades en energía y contenido frecuencial mediante procesamiento de señal digital avanzado

    Electric Network Frequency (ENF): Análisis de la huella de frecuencia de red eléctrica para detectar cambios de fuente o cortes temporales

    Detección Multi-Escala: Análisis simultáneo a diferentes resoluciones temporales (20ms, 50ms, 100ms)

    Clustering Inteligente: Agrupamiento temporal de detecciones usando DBSCAN para reducir falsos positivos

    Análisis de Armónicos: Correlación cruzada entre armónicos de la señal ENF para mayor robustez

    Umbrales Adaptativos: Sistemas de detección que se ajustan automáticamente al contenido del audio

Salidas Generadas

    Reporte Forense Completo: Archivo de texto con análisis detallado, explicaciones de métricas y recomendaciones
    Visualizaciones Gráficas (archivos):
        Espectrograma con marcas de anomalías detectadas
        Gráfico de energía suavizada con detecciones temporales
        Comparativa energía original vs. suavizada
        Análisis completo de señal ENF (cuando se solicita)
    Resultados Cuantitativos: Puntuaciones de confianza, estabilidad ENF, correlación de armónicos y métricas de calidad

Instalación
Requisitos del Sistema

    Python 3.8 o superior

    4GB RAM mínimo (8GB recomendado para archivos largos)

    500MB espacio en disco para dependencias


Dependencias

pip install librosa numpy matplotlib scipy scikit-learn tqdm


Instalación del Programa

# Descargar el script
wget https://github.com/tu-usuario/audio-forensic-analyzer/raw/main/audio_forensic_analyzer.py

# Dar permisos de ejecución (opcional)
chmod +x audio_forensic_analyzer.py

Uso
Ejemplos Básicos

# Análisis estándar
python audio_forensic_analyzer.py audio.wav

# Análisis completo con ENF
python audio_forensic_analyzer.py audio.wav --enf-analysis --multi-scale

# Máximo rigor para análisis forense
python audio_forensic_analyzer.py audio.wav --sensitivity 8.0 --enf-analysis --enf-fundamental 50

# Análisis rápido para verificación inicial
python audio_forensic_analyzer.py audio.wav --sensitivity 6.0 --frame 0.1


Parámetros Principales

    --sensitivity: Umbral de detección para cambios de energía (4.0-10.0, default: 6.0)

    --spectral: Sensibilidad para cambios espectrales (0.3-0.8, default: 0.4)

    --enf-analysis: Activa análisis de Electric Network Frequency

    --enf-fundamental: Frecuencia de red (50/60 Hz, default: 50)

    --multi-scale: Análisis multi-resolución temporal

    --auto-calibrate: Calibración automática de parámetros

    --output: Prefijo personalizado para archivos de salida

nterpretación de Resultados
Métricas de Confianza

Confianza General (0.0-1.0)

        0.8: Alta probabilidad de audio original

    0.6-0.8: Verificación manual recomendada

    < 0.6: Alta sospecha de manipulación

Estabilidad ENF (0.0-1.0)

        0.8: Excelente - señal continua y estable

    0.6-0.8: Moderada - variaciones naturales aceptables

    < 0.6: Baja - posible edición o múltiples fuentes

Correlación Armónicos (0.0-1.0)

        0.8: Fuerte - fuente eléctrica única

    0.5-0.8: Moderada - algunas discrepancias

    < 0.5: Débil - múltiples fuentes posibles

Tipos de Anomalías Detectadas

    Inicio/Fin de Grabación: Anomalías en extremos temporales (<10% o >90%)

    Transiciones de Contenido: Cambios en medio de la grabación

    Saltos ENF: Discontinuidades en frecuencia de red

    Clusters Temporales: Múltiples detecciones cercanas que sugieren región problemática

Metodología de Verificación Manual
Escucha Activa

Procedimiento Recomendado:

    Preparación: Usar auriculares de alta calidad en ambiente silencioso

    Enfoque Segmentado: Escuchar 10 segundos antes y después de cada anomalía detectada

    Búsqueda de Artefactos:

        Clicks o pops audibles

        Cambios abruptos en reverberación

        Discontinuidades en ruido de fondo

        Variaciones inusuales en calidad vocal

    Comparativa A/B: Contrastar segmentos sospechosos con partes claramente originales

Herramientas de Software Libre Recomendadas

Análisis Espectral Avanzado:

    Audacity: Análisis espectral, filtros y visualización de waveform

    Sonic Visualiser: Análisis espectral avanzado con capas

    Spek: Generación rápida de espectrogramas

    QJackCtl: Conexión de herramientas de audio profesional

Análisis Forense Especializado:

    Wav2PNG: Generación de espectrogramas de alta resolución

    Sox: Procesamiento por línea de comandos con análisis estadístico

    EchoSleuth: Detección de ecos y reverberaciones artificiales

Validación de Metadatos:

    MediaInfo: Análisis exhaustivo de metadatos técnicos

    ExifTool: Lectura y validación de metadatos de archivos

    FFmpeg: Análisis profundo de estructura de archivos

Técnicas de Análisis Complementarias

Análisis de Metadatos:

    Verificar consistencia de timestamps

    Analizar historial de software de edición

    Validar codecs y tasas de compresión

Análisis Estadístico:

    Distribución de amplitudes

    Análisis de ruido residual

    Patrones de compresión

Análisis de Contenido:

    Coherencia temporal de eventos

    Consistencia acústica del ambiente

    Patrones de habla y entonación

Descargo de Responsabilidades
Limitaciones Técnicas

Este software es una herramienta de asistencia al análisis forense y no debe considerarse como prueba definitiva de autenticidad o manipulación de audio. Las detecciones automáticas deben ser validadas siempre mediante:

    Verificación Humana: Análisis experto por profesionales calificados

    Corroboración Multi-herramienta: Uso de múltiples softwares de análisis

    Contexto Investigativo: Consideración del caso específico y circunstancias

Limitaciones Conocidas

    Ediciones Profesionales: No detecta ediciones con crossfades suaves o procesamiento profesional avanzado

    Condiciones de Grabación: Variaciones naturales en ambiente acústico pueden generar falsos positivos

    Calidad de Audio: Archivos muy comprimidos o con mucho ruido reducen la efectividad

    ENF Limitado: Requiere que el audio contenga señal de red eléctrica detectable

Uso Ético y Legal

El usuario es responsable de:

    Obtener autorización legal para analizar los archivos de audio

    Cumplir con las leyes locales de privacidad y evidencia digital

    Utilizar los resultados dentro del marco legal aplicable

    Documentar la cadena de custodia de las evidencias

    No utilizar el software para actividades ilegales o no éticas

Recomendaciones para Investigaciones Profundas
Protocolo de Análisis Completo

Fase 1: Análisis Automático Inicial

    Ejecutar este analizador con parámetros conservadores

    Generar todas las visualizaciones disponibles

    Documentar hallazgos iniciales

Fase 2: Verificación Manual

    Escucha activa de segmentos sospechosos

    Análisis espectral con múltiples herramientas

    Validación de metadatos técnicos

Fase 3: Análisis Especializado

    Búsqueda de patrones de compresión inconsistentes

    Análisis de reverberación y acústica ambiental

    Estudio de firmas digitales de dispositivos

Fase 4: Corroboración

    Contrastar resultados con otras herramientas

    Consultar con múltiples expertos si es posible

    Documentar metodología y hallazgos exhaustivamente

Casos de Uso Específicos

Forense Digital:

    Ejecutar con máxima sensibilidad (--sensitivity 10.0)

    Activar todos los análisis disponibles

    Realizar múltiples pasadas con diferentes parámetros

    Documentar proceso para cadena de custodia

Verificación Periodística:

    Balance entre sensibilidad y especificidad

    Focus en detección de cortes evidentes

    Análisis rápido para verificación inicial

Control de Calidad:

    Parámetros adaptados al tipo de contenido

    Enfoque en detección de errores técnicos

    Análisis comparativo entre versiones

Licencia

Copyright (c) 2025 David Casas M. - Competencia_Digital

Este programa es software libre: usted puede redistribuirlo y/o modificarlo bajo los 
términos de la Licencia Pública General GNU publicada por la Free Software Foundation, 
ya sea la versión 3 de la Licencia, o (a su elección) cualquier versión posterior.

Este programa se distribuye con la esperanza de que sea útil, pero 
SIN NINGUNA GARANTÍA; incluso sin la garantía implícita de COMERCIALIZACIÓN o IDONEIDAD PARA UN PROPÓSITO PARTICULAR. 
Consulte la Licencia Pública General GNU para más detalles.

Debe haber recibido una copia de la Licencia Pública General GNU junto con este programa. 
Si no ha sido así, consulte https://www.gnu.org/licenses/.
Atribuciones

Creador: David Casas M. - Competencia_Digital
Asistencia IA: DEEPSEEK
Licencia: GPL-3.0
YouTube: https://youtube.com/@competencia_digital
Sitio Web: https://competenciadigital.name
Agradecimientos

Este software utiliza las siguientes bibliotecas de código abierto:

    Librosa para análisis y procesamiento de audio

    NumPy y SciPy para computación científica

    Matplotlib para visualizaciones

    Scikit-learn para algoritmos de machine learning

    TQDM para barras de progreso

Soporte y Contribuciones

Para reportar issues o sugerir mejoras, por favor utilizar el sistema de issues 
del repositorio oficial. Las contribuciones son bienvenidas mediante pull requests.

Canales de Soporte:

    Documentación oficial: [Enlace a documentación]

    Comunidad: [Enlace a foro/comunidad]

    Contacto profesional: [Email de contacto]

Changelog
Versión 1.0.0

    Análisis espectral y energético tradicional

    Detección de Electric Network Frequency (ENF)

    Sistema multi-escala temporal

    Clustering inteligente de anomalías

    Reportes explicativos detallados

    Visualizaciones completas del análisis


