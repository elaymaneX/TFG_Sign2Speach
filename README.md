# Sign2Speach – Traducción de Lengua de Signos a Voz mediante Computer Vision y Edge Computing

Este proyecto es el Trabajo de Fin de Grado (TFG) del grado en Ingeniería Informática en la Universitat Politècnica de València – Campus de Alcoy.

## 🎯 Objetivo

Desarrollar un sistema basado en **computer vision** que permita traducir secuencias de lengua de signos americana (ASL) en texto y posteriormente en voz, utilizando tecnologías de **edge computing** como la Coral TPU de Google.

## 🧠 Componentes del proyecto

### 1. `Entrenamiento/`
Código para:
- Preprocesamiento y extracción de landmarks con MediaPipe.
- Entrenamiento del modelo secuencial.
- Conversión del modelo a TFLite.

### 2. `sign2speech_app/`
Aplicación desarrollada con **PyQt5** para:
- Capturar signos en tiempo real.
- Predecir palabras con el modelo TFLite.
- Formar frases con un LLM local.
- Convertir texto a voz (TTS).

### 3. `Codigo de Edge Tpu/`
Scripts y modelo optimizado (`model_edgetpu.tflite`) para ejecutar inferencia directamente en una Coral TPU conectada a servicios públicos u otros dispositivos embebidos.

