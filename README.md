# Comparación de Detectores y Reconocedores de Matrículas Vehiculares 🚗📸

Este proyecto presenta un estudio comparativo de diversos enfoques para la **detección y reconocimiento automático de matrículas vehiculares**, integrando tanto **métodos clásicos de visión por computador** como **modelos modernos de aprendizaje profundo**.  
El objetivo principal es evaluar la eficacia, precisión y eficiencia temporal de cada técnica en distintas condiciones visuales, con el propósito de determinar qué método ofrece un rendimiento más robusto y generalizable.

---

## 🧠 Introducción

La identificación automática de matrículas constituye una aplicación esencial en ámbitos como el control de acceso vehicular, la gestión de tráfico, la seguridad urbana y los sistemas de peaje automatizados.  
A lo largo de los años, las técnicas empleadas han evolucionado desde algoritmos basados en procesamiento morfológico hasta arquitecturas profundas de redes neuronales convolucionales (CNN).

Este trabajo implementa y compara cuatro métodos representativos, con el fin de establecer un marco experimental reproducible y contrastar los resultados bajo métricas objetivas.

---

## 🧩 Métodos Comparados

1. **Detección por contornos (OpenCV)**  
   Método tradicional que utiliza operaciones morfológicas, umbralización adaptativa y filtrado por proporciones geométricas para localizar regiones candidatas a matrículas.  
   Es una técnica eficiente, pero sensible a variaciones de iluminación y ángulos de captura.

2. **EasyOCR**  
   Sistema OCR preentrenado capaz de detectar y reconocer texto directamente sobre las imágenes.  
   Permite una implementación sencilla y resultados razonables sin necesidad de entrenamiento adicional.

3. **YOLOv8 preentrenado (Ultralytics)**  
   Detector de objetos de última generación que ofrece un excelente equilibrio entre velocidad y precisión.  
   Se emplea un modelo preentrenado en COCO, ajustado para la detección de matrículas mediante transferencia de aprendizaje.

4. **CNN Personalizada (PyTorch)**  
   Arquitectura desarrollada desde cero para la tarea de detección y reconocimiento, utilizando un conjunto de datos reducido con fines experimentales.  
   Este enfoque permite un mayor control sobre las capas y los hiperparámetros, facilitando el análisis comparativo del desempeño.

---

## ⚙️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.10+
- **Bibliotecas principales:**  
  - OpenCV  
  - EasyOCR  
  - Ultralytics YOLOv8  
  - PyTorch  
  - NumPy, Pandas, Matplotlib  

---

## 📊 Metodología y Evaluación

El análisis comparativo se realiza bajo tres dimensiones principales:

1. **Precisión y Recuperación (Detección de Matrículas):**  
   Evalúa la capacidad del modelo para identificar correctamente las regiones que contienen matrículas.

2. **Exactitud del OCR (Reconocimiento de Texto):**  
   Compara el texto reconocido con el texto real, considerando errores de carácter y palabras.

3. **Tiempo de Inferencia:**  
   Mide la eficiencia de procesamiento de cada modelo, en segundos por imagen.

El conjunto de pruebas incluye imágenes con diferentes resoluciones, iluminaciones, ángulos y condiciones ambientales, con el propósito de simular escenarios reales.

---

## 🚀 Ejecución del Proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/dvalloz/Comparacion-de-detectores-de-matriculas.git
   cd license-plate-detection-comparison
