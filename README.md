# Reconocimiento de Gestos con IA
## Sistema de Interacción Visual en Tiempo Real

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)

---

## 📋 Descripción

Sistema avanzado de reconocimiento de gestos de mano que utiliza **MediaPipe** y **Machine Learning** para aplicar efectos visuales en tiempo real sobre el cuerpo del usuario. El proyecto ha evolucionado desde un clasificador de gestos básico hacia una plataforma completa de interacción visual.

### Características Principales

✅ **Detección de gestos de mano** con MediaPipe Hand Landmarker  
✅ **Detección de pose del torso superior** con MediaPipe Pose  
✅ **Clasificador ML** entrenado con Random Forest y MLP  
✅ **5 efectos visuales** aplicados sobre el usuario en tiempo real  
✅ **Transiciones suaves** con sistema de fade automático  
✅ **Modo de captura de dataset** para entrenar modelos personalizados  
✅ **Arquitectura modular** fácil de extender  

---

## 📁 Estructura del Proyecto

```
reconocimiento-IA/
│
├── src/                          # Código fuente principal
│   ├── core/                     # Módulos de detección
│   │   ├── gesture_detector.py  # Detección de gestos de mano
│   │   ├── pose_detector.py     # Detección de pose corporal
│   │   └── gesture_classifier.py # Clasificador ML
│   │
│   ├── visual/                   # Módulos de efectos visuales
│   │   ├── body_regions.py      # Cálculo de regiones corporales
│   │   ├── visual_effects.py    # Biblioteca de efectos
│   │   ├── effect_renderer.py   # Renderizado suavizado
│   │   └── image_manager.py     # Gestión de imágenes
│   │
│   └── data/                     # Módulos de datos
│       ├── dataset_collector.py # Captura de landmarks
│       └── dataset_stats.py     # Análisis de dataset
│
├── scripts/                      # Scripts de utilidad
│   ├── train_model.py           # Entrenar clasificador ML
│   ├── setup_images.py          # Generar imágenes placeholder
│   └── test_imports.py          # Verificar instalación
│
├── models/                       # Modelos de MediaPipe
│   ├── hand_landmarker.task     # Modelo de mano (descargado)
│   └── pose_landmarker_lite.task # Modelo de pose (descargado)
│
├── dataset/                      # Datasets generados
│   ├── dataset_landmarks.csv    # Landmarks capturados
│   ├── gesture_model.pkl        # Modelo entrenado
│   └── label_encoder.pkl        # Codificador de etiquetas
│
├── images/                       # Imágenes de gestos
│   ├── gesture_1.jpg
│   ├── gesture_2.jpg
│   └── ...
│
├── docs/                         # Documentación
│   └── README_EFECTOS.md        # Guía de efectos visuales
│
├── main.py                       # Punto de entrada principal
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

---

## 🚀 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Javotass/Reconocimiento-IA.git
cd Reconocimiento-IA
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Verificar Instalación

```bash
python scripts/test_imports.py
```

---

## 🎮 Uso

### Modo LIVE (Interacción Visual)

```bash
python main.py
```

El sistema iniciará en modo LIVE donde los gestos activan efectos visuales en tiempo real.

**Controles principales:**
- **Q** - Salir
- **P** - Mostrar/ocultar panel lateral
- **D** - Activar modo debug
- **B** - Mostrar regiones corporales (debug)
- **S** - Capturar pantalla

### Modo RECORD (Captura de Dataset)

```bash
python main.py
# Presionar M para cambiar a modo RECORD
```

**Controles de grabación:**
- **1/2/3/4/O** - Grabar gesto durante 5 segundos
- **F** - Guardar dataset CSV
- **M** - Volver a modo LIVE

### Entrenar Modelo Personalizado

```bash
# 1. Capturar datos en modo RECORD
# 2. Entrenar modelo
python scripts/train_model.py

# 3. El modelo se guarda en dataset/gesture_model.pkl
```

### Generar Imágenes Placeholder

```bash
python scripts/setup_images.py
```

---

## 🎨 Efectos Visuales

| Gesto | Nombre | Descripción |
|-------|--------|-------------|
| **1 dedo** | Energy Aura | Aura cian brillante alrededor del cuerpo |
| **2 dedos** | Armor Overlay | Armadura metálica sobre el torso |
| **3 dedos** | Fire Shoulders | Llamas naranjas en los hombros |
| **4 dedos** | Ice Shield | Escudo cristalino congelado |
| **OK** | Golden Crown | Corona dorada sobre la cabeza |

Ver documentación completa en [`docs/README_EFECTOS.md`](docs/README_EFECTOS.md)

---

## 🔧 Tecnologías

- **Python 3.8+**
- **OpenCV** - Procesamiento de imágenes
- **MediaPipe** - Detección de mano y pose
- **scikit-learn** - Machine Learning
- **NumPy** - Operaciones numéricas
- **pandas** - Manejo de datos

---

## 📊 Arquitectura

```
┌──────────────────────────────────────────────────┐
│                    main.py                       │
│         (Orquestador principal)                  │
└──────────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  gesture_detect │     │  pose_detector   │
│     (entrada)   │     │  (cuerpo)        │
└─────────────────┘     └─────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
          ┌─────────────────────┐
          │   body_regions      │
          │  (áreas corporales) │
          └─────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ visual_effects  │────▶│ effect_renderer │
│   (biblioteca)  │     │   (suavizado)   │
└─────────────────┘     └─────────────────┘
```

---

## 🎓 Progresión del Proyecto

1. **Fase 1:** Detección básica de gestos con reglas geométricas
2. **Fase 2:** Captura de dataset con landmarks normalizados
3. **Fase 3:** Entrenamiento de clasificador ML (RF + MLP)
4. **Fase 4:** Sistema de interacción visual con efectos sobre el cuerpo

---

## 📝 Uso Académico

Este proyecto demuestra:

- Integración de múltiples modelos de MediaPipe
- Pipeline completo de Machine Learning
- Arquitectura modular y escalable
- Interacción humano-computador en tiempo real
- Procesamiento de video con efectos visuales

---

## 🔮 Mejoras Futuras

- [ ] Más efectos visuales (partículas, animaciones)
- [ ] Segmentación avanzada del cuerpo
- [ ] Efectos combinados (múltiples gestos)
- [ ] Cambio de fondo en tiempo real
- [ ] Exportación de video con efectos

---

## 📄 Licencia

Este proyecto es de código abierto bajo licencia MIT.

---

## 👤 Autor

**Javier** - [Javotass](https://github.com/Javotass)

---

## 🙏 Agradecimientos

- [MediaPipe](https://google.github.io/mediapipe/) por los modelos de detección
- [OpenCV](https://opencv.org/) por las herramientas de visión
- La comunidad de Python por las excelentes librerías
