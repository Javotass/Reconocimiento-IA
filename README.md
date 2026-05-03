# Reconocimiento de Gestos con IA

Sistema en tiempo real que detecta gestos de tu mano a través de la cámara web y aplica efectos visuales espectaculares sobre tu cuerpo. Muestra 1, 2, 3, 4 o 5 dedos y verás cómo aparece una armadura, fuego, confeti o una corona dorada sobre ti.

---

## ¿Qué hace este programa?

1. **Abre tu cámara web** y te muestra en pantalla.
2. **Detecta tu mano** en tiempo real usando inteligencia artificial.
3. **Reconoce cuántos dedos estás mostrando** (del 1 al 5).
4. **Aplica un efecto visual** distinto según el gesto detectado.

Todo ocurre de forma instantánea, sin retardo perceptible.

---

## Efectos visuales disponibles

| Gesto | Dedos | Nombre del efecto | ¿Qué se ve? |
|-------|-------|-------------------|-------------|
| ☝️ | 1 dedo | **Shadow Mode** | Armadura de Netherite superpuesta sobre tu torso |
| ✌️ | 2 dedos | **Golden Hour** | Filtro dorado cinematográfico sobre toda la imagen |
| 🤟 | 3 dedos | **Fire Shoulders** | Llamas animadas ardiendo en tus hombros |
| 🖐️ | 4 dedos | **Confetti Rain** | Lluvia de confeti de colores cayendo sobre ti |
| 🖐️ | 5 dedos | **Golden Crown** | Corona dorada flotando sobre tu cabeza |

---

## Requisitos del sistema

- **Python 3.8 – 3.11** (MediaPipe aún no es compatible con Python 3.12+)
- **Cámara web** (integrada o externa)
- **Windows / macOS / Linux**
- Conexión a internet la primera vez (para descargar los modelos de IA)

---

## Instalación paso a paso

### 1. Descarga el proyecto

```bash
git clone https://github.com/Javotass/Reconocimiento-IA.git
cd "reconocimiento IA"
```

### 2. Crea un entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

| Librería | Para qué sirve |
|----------|----------------|
| `opencv-python` | Captura y procesamiento de vídeo |
| `mediapipe` | Detección de manos y pose corporal (IA de Google) |
| `numpy` | Cálculos matemáticos rápidos |
| `scikit-learn` | Modelo de machine learning para clasificar gestos |
| `pandas` | Manejo de datos de entrenamiento |
| `joblib` | Guardar y cargar el modelo entrenado |

### 4. Verifica que todo esté bien

```bash
python scripts/test_imports.py
```

---

## Cómo usar el programa

### Ejecutar

```bash
python main.py
```

Se abrirá una ventana con tu cámara. Pon tu mano delante y muestra diferentes números de dedos para ver los efectos.

### Controles de teclado

| Tecla | Acción |
|-------|--------|
| `Q` | Salir del programa |
| `M` | Cambiar entre modo **EN VIVO** y modo **GRABACIÓN** |
| `D` | Mostrar/ocultar información de depuración (confianza del gesto) |
| `P` | Mostrar/ocultar el panel lateral con imágenes de gestos |
| `B` | Ver las regiones corporales detectadas |
| `S` | Tomar una captura de pantalla (se guarda en `capturas/`) |
| `R` | Recargar las imágenes de efectos |

---

## Estructura del proyecto

```
reconocimiento IA/
│
├── main.py                    ← Punto de entrada principal
├── requirements.txt           ← Lista de dependencias
│
├── src/
│   ├── core/                  ← Núcleo de detección con IA
│   │   ├── gesture_detector.py    (detecta la mano y los dedos)
│   │   ├── pose_detector.py       (detecta el cuerpo: hombros, cabeza...)
│   │   └── gesture_classifier.py  (clasifica el gesto con ML)
│   │
│   └── visual/                ← Todo lo visual
│       ├── filters/           ← Los 5 efectos visuales
│       │   ├── shadow_mode.py
│       │   ├── golden_hour.py
│       │   ├── fire_shoulders.py
│       │   ├── confetti_rain.py
│       │   └── golden_crown.py
│       ├── body_regions.py        (calcula dónde están hombros, torso, cabeza)
│       ├── visual_effects.py      (conecta gestos con efectos)
│       └── effect_renderer.py     (renderiza con transiciones suaves)
│
├── scripts/
│   ├── train_model.py         ← Entrena el modelo con tus datos
│   ├── setup_images.py        ← Genera imágenes de prueba
│   └── test_imports.py        ← Verifica que la instalación es correcta
│
├── models/                    ← Modelos de IA preentrenados (MediaPipe)
│   ├── hand_landmarker.task
│   └── pose_landmarker_lite.task
│
├── dataset/                   ← Datos de entrenamiento y modelo guardado
│   ├── dataset_landmarks.csv      (tus datos grabados)
│   ├── gesture_model.pkl          (modelo entrenado)
│   └── label_encoder.pkl
│
├── images/                    ← Imágenes usadas en los efectos
└── capturas/                  ← Capturas de pantalla guardadas con S
```

---

## Cómo funciona la IA por dentro

El programa usa **dos capas de inteligencia artificial**:

### Capa 1 — Detección visual (MediaPipe, de Google)
- **Detector de mano:** Localiza 21 puntos clave de tu mano (nudillos, puntas de dedos, etc.).
- **Detector de pose:** Localiza 33 puntos de tu cuerpo (hombros, caderas, cabeza...).

### Capa 2 — Clasificación del gesto
1. **Método geométrico (siempre activo):** Compara la posición de la punta de cada dedo con su articulación para saber si está extendido o doblado.
2. **Modelo de Machine Learning (si está entrenado):** Un clasificador Random Forest + red neuronal entrenado con tus propios gestos.

Si el modelo entrenado no está seguro (confianza menor al 55%), el programa vuelve automáticamente al método geométrico.

---

## Entrenar tu propio modelo (opcional)

Si los gestos no se detectan bien con tu mano, puedes entrenar el modelo con tus propios datos en tres pasos:

### Paso 1 — Grabar tus gestos

```bash
python main.py
```

Presiona `M` para entrar en **modo GRABACIÓN**. Luego usa estas teclas para grabar 5 segundos de cada gesto:

| Tecla | Gesto que graba |
|-------|----------------|
| `1` | 1 dedo |
| `2` | 2 dedos |
| `3` | 3 dedos |
| `4` | 4 dedos |
| `O` | 5 dedos (mano abierta) |

Repite varias veces cada gesto para tener más datos. Presiona `F` para guardar.

### Paso 2 — Entrenar el modelo

```bash
python scripts/train_model.py
```

El script entrena el modelo y lo guarda en `dataset/gesture_model.pkl`. La próxima vez que ejecutes el programa, usará automáticamente tu modelo personalizado.

---

## Solución de problemas comunes

**La cámara no se abre**
- Comprueba que ninguna otra aplicación esté usando la cámara.
- Prueba cambiando `CAMERA_INDEX = 0` a `1` en `main.py` si tienes varias cámaras.

**No detecta mi mano**
- Asegúrate de tener buena iluminación.
- Pon la mano a una distancia de 30–80 cm de la cámara.
- Un fondo claro mejora la detección.

**Los efectos van lentos**
- Reduce la resolución en `main.py`: cambia `CAMERA_WIDTH` y `CAMERA_HEIGHT` a valores menores (ej: 480×360).

**Error al instalar MediaPipe**
- Asegúrate de usar Python 3.8–3.11. MediaPipe aún no es compatible con Python 3.12+.

---

## Tecnologías usadas

- **[MediaPipe](https://mediapipe.dev/)** — Framework de IA de Google para detección de manos y cuerpo
- **[OpenCV](https://opencv.org/)** — Procesamiento de vídeo en tiempo real
- **[scikit-learn](https://scikit-learn.org/)** — Algoritmos de machine learning
- **Python 3** — Lenguaje de programación

---

## Autor

Desarrollado por **[Javotass](https://github.com/Javotass)**
