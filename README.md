# Reconocimiento de Gestos con Inteligencia Artificial

Sistema de visión por computador en tiempo real que detecta gestos de la mano mediante la cámara web y proyecta efectos visuales sobre el cuerpo del usuario. Combina detección de landmarks con MediaPipe y un clasificador Random Forest entrenado con datos propios.

---

## Demostración

| Gesto | Efecto |
|-------|--------|
| 1 dedo | **Shadow Mode** — armadura de Netherite superpuesta sobre el torso |
| 2 dedos | **Golden Hour** — filtro cinematográfico cálido sobre toda la imagen |
| 3 dedos | **Fire Shoulders** — llamas animadas en los hombros |
| 4 dedos | **Confetti Rain** — lluvia de confeti de colores |
| OK (pulgar + índice) | **Golden Crown** — corona dorada sobre la cabeza |

---

## Arquitectura del sistema

El sistema se organiza en tres capas independientes:

```
Cámara → MediaPipe Hands → GestureClassifier → EffectRenderer → Pantalla
                   ↓
         MediaPipe Pose → BodyRegions → (posicionamiento de efectos)
```

### 1. Detección de landmarks (MediaPipe)

- **HandLandmarker**: localiza 21 puntos clave de la mano (puntas, nudillos, muñeca). Funciona a ~30 FPS sobre CPU sin GPU dedicada.
- **PoseLandmarker Lite**: localiza 33 puntos del cuerpo (hombros, caderas, nariz). Se usa exclusivamente para anclar los efectos visuales en la posición correcta del usuario, no para clasificar gestos.

### 2. Clasificación del gesto

El `GestureDetector` combina dos métodos:

**Método geométrico** (activo siempre como fallback):
- Compara la coordenada Y de la punta de cada dedo con su articulación PIP. Si la punta está por encima, el dedo se considera extendido.
- Para el gesto OK detecta la distancia euclidiana entre la punta del pulgar y del índice.

**Modelo de Machine Learning** (prioritario cuando está disponible):
- Random Forest de 300 árboles entrenado con datos propios.
- Entrada: 63 valores (21 landmarks × 3 coordenadas), normalizados por traslación y escala invariante a la distancia de la mano.
- Salida: clase entre {ONE, TWO, THREE, FOUR, OK}.
- Umbral de confianza: 55%. Por debajo de ese valor el sistema cae al método geométrico.

**Suavizado temporal**: los últimos 10 resultados se almacenan en un buffer circular. Se devuelve el gesto mayoritario solo si supera el 50% del buffer, eliminando parpadeos.

### 3. Renderizado de efectos

`EffectRenderer` gestiona las transiciones entre efectos con fade-in/fade-out configurable (~12 frames de transición). Cada filtro recibe el frame de OpenCV y el objeto `BodyRegions` con las coordenadas en píxeles de las regiones corporales (hombros, torso, cabeza).

---

## Resultados del entrenamiento

Dataset recopilado manualmente con 11.658 muestras totales:

| Gesto | Muestras |
|-------|----------|
| ONE   | 2.203    |
| TWO   | 2.277    |
| THREE | 2.509    |
| FOUR  | 2.464    |
| OK    | 2.205    |

Antes del entrenamiento el dataset se balancea por undersampling (todas las clases al mínimo: 2.203). La división es 70% entrenamiento / 15% validación / 15% test, estratificada.

**Random Forest — 99,2% de accuracy en test**

```
              precision    recall  f1-score   support
        FOUR      0.991     1.000     0.995       331
          OK      0.991     1.000     0.995       330
         ONE      0.994     0.991     0.992       331
       THREE      1.000     0.976     0.988       331
         TWO      0.985     0.994     0.989       330
    accuracy                          0.992      1653
```

Matriz de confusión:

```
         FOUR    OK   ONE THREE   TWO
FOUR      331     0     0     0     0
OK          0   330     0     0     0
ONE         0     0   328     0     3
THREE       3     3     0   323     2
TWO         0     0     2     0   328
```

El gesto THREE es el más problemático (8 errores): se confunde con FOUR (3 casos) y con OK (3 casos), probablemente porque la posición de los dedos en ambos casos es geométricamente similar en el plano 2D de la cámara. FOUR y OK tuvieron clasificación perfecta (0 errores).

El MLP obtuvo 98,5%, por lo que el modelo guardado es el Random Forest.

---

## Requisitos

- **Python 3.8–3.11** (MediaPipe no es compatible con 3.12+)
- Cámara web (integrada o externa)
- Conexión a internet la primera vez (descarga automática de los modelos de MediaPipe)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Javotass/Reconocimiento-IA.git
cd "reconocimiento IA"

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python scripts/test_imports.py
```

---

## Uso

### Modo EN VIVO

```bash
python main.py
```

La ventana muestra la imagen de la cámara con los landmarks de la mano dibujados. A la derecha aparece un panel con el gesto activo. Los efectos se renderizan sobre el frame cuando el sistema detecta pose corporal.

### Controles

| Tecla | Acción |
|-------|--------|
| `Q` / `Esc` | Salir |
| `M` | Alternar entre modo LIVE y modo RECORD |
| `D` | Mostrar información de debug (confianza, estado de la pose, barra de intensidad) |
| `P` | Mostrar / ocultar el panel lateral |
| `B` | Visualizar las regiones corporales detectadas (debug) |
| `S` | Guardar captura de pantalla en `capturas/` |
| `R` | Recargar imágenes desde disco |

---

## Entrenar el modelo con datos propios

Si los gestos no se reconocen bien con una mano concreta, es posible reentrenar con datos nuevos en dos pasos.

### Paso 1 — Grabar datos (modo RECORD)

```bash
python main.py
```

Pulsar `M` para entrar en modo RECORD. Las teclas `1` `2` `3` `4` `O` inician una sesión de grabación de 5 segundos para cada gesto. Pulsar `F` para guardar el CSV sin cerrar el programa.

Se recomienda grabar al menos 200 muestras por gesto, variando la inclinación y la distancia a la cámara. Para ver cuántas muestras hay acumuladas:

```bash
python src/data/dataset_stats.py
```

### Paso 2 — Entrenar

```bash
python scripts/train_model.py
```

El script limpia el dataset, balancea las clases, entrena Random Forest y MLP, imprime las métricas detalladas y guarda el mejor modelo en `dataset/gesture_model.pkl`.

---

## Estructura del proyecto

```
reconocimiento IA/
│
├── main.py                        # Bucle principal y composición de ventana
├── requirements.txt
│
├── src/
│   ├── core/
│   │   ├── gesture_detector.py    # Detección de mano + suavizado + clasificación
│   │   ├── gesture_classifier.py  # Wrapper del modelo sklearn
│   │   └── pose_detector.py       # Detección de pose corporal con suavizado EMA
│   │
│   └── visual/
│       ├── filters/
│       │   ├── shadow_mode.py     # Efecto 1: armadura Netherite (polígono procedural + textura PNG)
│       │   ├── golden_hour.py     # Efecto 2: corrección de color + viñeta
│       │   ├── fire_shoulders.py  # Efecto 3: llamas animadas con jitter sinusoidal
│       │   ├── confetti_rain.py   # Efecto 4: sistema de partículas con rotación
│       │   └── golden_crown.py    # Efecto 5: PNG con alpha anclado por FaceMesh
│       ├── body_regions.py        # Cálculo de regiones corporales en píxeles
│       ├── visual_effects.py      # Despacho gesto → función de efecto
│       └── effect_renderer.py     # Fade-in/out entre efectos
│
├── scripts/
│   ├── train_model.py             # Entrenamiento RF + MLP con métricas
│   ├── setup_images.py            # Genera imágenes de placeholder
│   └── test_imports.py            # Verificación de dependencias
│
├── dataset/
│   ├── dataset_landmarks.csv      # 11.658 muestras (63 features + etiqueta)
│   ├── gesture_model.pkl          # Modelo entrenado (Random Forest)
│   └── label_encoder.pkl          # Codificador de etiquetas
│
├── models/                        # Modelos de MediaPipe (descarga automática)
│   ├── hand_landmarker.task
│   └── pose_landmarker_lite.task
│
├── images/                        # Texturas PNG para los efectos
└── capturas/                      # Capturas guardadas con la tecla S
```

---

## Tecnologías utilizadas

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| `mediapipe` | 0.10.0 | Detección de mano (HandLandmarker) y cuerpo (PoseLandmarker) |
| `opencv-python` | 4.8.0 | Captura de vídeo, rendering y operaciones de imagen |
| `scikit-learn` | 1.3.0 | RandomForestClassifier y MLPClassifier |
| `numpy` | 1.24.0 | Operaciones matriciales y blend de píxeles |
| `pandas` | 2.0.0 | Carga y preprocesado del dataset CSV |
| `joblib` | 1.3.0 | Serialización del modelo entrenado |

---

## Solución de problemas

**La cámara no se abre**
Cambiar `CAMERA_INDEX = 0` a `1` (o `2`) en `main.py` si hay varias cámaras conectadas.

**No detecta la mano**
Iluminación frontal uniforme y fondo no saturado mejoran notablemente la detección. Distancia recomendada: 30–80 cm.

**Los efectos no aparecen aunque detecta la mano**
Los efectos requieren que el sistema detecte también la pose corporal (torso visible). El indicador de estado de pose aparece en pantalla si el modo debug (tecla `D`) está activo.

**Error de encoding al ejecutar `train_model.py` en Windows**
```bash
$env:PYTHONIOENCODING="utf-8"; python scripts/train_model.py
```

---

## Autor

Desarrollado por **[Javotass](https://github.com/Javotass)**
