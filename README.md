# Comparación de detectores de matrículas 🚗📸

Este proyecto compara distintos métodos de **detección y reconocimiento de matrículas** de vehículos, utilizando desde técnicas clásicas de visión por computador hasta modelos de deep learning.

## 🧩 Métodos comparados
1. **Detección por contornos (OpenCV)** – Enfoque basado en morfología y filtrado por proporciones.
2. **EasyOCR** – OCR preentrenado capaz de detectar y reconocer texto en imágenes.
3. **YOLOv8 preentrenado** – Detector de objetos avanzado para localizar matrículas.
4. **CNN personalizada (PyTorch)** – Red neuronal convolucional desarrollada desde cero.

## ⚙️ Tecnologías usadas
- Python 3.10+
- OpenCV
- EasyOCR
- Ultralytics YOLOv8
- PyTorch
- Matplotlib, NumPy, Pandas

## 📊 Evaluación
Los métodos se comparan en base a:
- **Precisión** y **recuperación** (detección de matrículas)
- **Exactitud OCR** (texto reconocido)
- **Tiempo de inferencia**

## 🚀 Ejecución
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tuusuario/license-plate-detection-comparison.git
   cd license-plate-detection-comparison
