# Sistema de Reconocimiento Gestual con Efectos Visuales

## 🎯 Evolución del Proyecto

Este proyecto ha evolucionado desde un **sistema de reconocimiento de gestos** hacia una **plataforma de interacción visual en tiempo real** que aplica transformaciones visuales sobre el usuario basándose en gestos de mano detectados.

### Cambios Principales

1. **Mantiene** el reconocimiento gestual como mecanismo de control
2. **Añade** detección de pose del torso superior (MediaPipe Pose)
3. **Aplica** efectos visuales directamente sobre el cuerpo del usuario
4. **Elimina** (opcionalmente) el panel lateral de imágenes para enfoque en efectos

## 🚀 Arquitectura Modular

```
┌─────────────────────┐
│   main.py           │  ← Orquestador principal
├─────────────────────┤
│ gesture_detector.py │  ← Detección de gestos (entrada)
│ pose_detector.py    │  ← Detección de pose (torso superior)
├─────────────────────┤
│ body_regions.py     │  ← Cálculo de áreas corporales
│ visual_effects.py   │  ← Biblioteca de efectos visuales
│ effect_renderer.py  │  ← Renderizado suavizado
└─────────────────────┘
```

## 🎨 Efectos Disponibles

Cada gesto activa un efecto visual diferente sobre el usuario:

| Gesto | Efecto | Descripción |
|-------|--------|-------------|
| **1 (UNO)** | Energy Aura | Aura cian brillante alrededor del cuerpo |
| **2 (DOS)** | Armor Overlay | Placa de armadura metálica sobre el torso |
| **3 (TRES)** | Fire Shoulders | Llamas naranjas sobre ambos hombros |
| **4 (FOUR)** | Ice Shield | Escudo congelado cristalino en el torso |
| **5 (OK)** | Golden Crown | Corona dorada flotando sobre la cabeza |

## 🎮 Controles Actualizados

### Controles Generales
- **Q** - Salir del programa
- **D** - Activar/desactivar overlay de debug
- **P** - Mostrar/ocultar panel lateral (imagen)
- **B** - Mostrar/ocultar regiones corporales (debug pose)
- **S** - Captura de pantalla
- **R** - Recargar imágenes desde disco

### Modo LIVE
El modo por defecto. Los gestos activan efectos visuales sobre el usuario.

### Modo RECORD
- **M** - Cambiar a modo RECORD (captura de dataset)
- **1/2/3/4/O** - Grabar gesto durante 5 segundos
- **F** - Forzar guardado del CSV
- **M** - Volver a modo LIVE

## 📋 Cómo Usar

### 1. Instalación
```bash
pip install -r requirements.txt
```

### 2. Ejecución Básica
```bash
python main.py
```

### 3. Uso Recomendado

**Para demostración de efectos visuales:**
1. Ejecutar `python main.py`
2. Presionar **P** para ocultar el panel lateral
3. Posicionarse sentado frente a la cámara
4. Realizar gestos con la mano para activar efectos
5. Los efectos aparecen directamente sobre tu cuerpo

**Para modo debug:**
1. Presionar **B** para ver las regiones corporales detectadas
2. Presionar **D** para ver información de confianza y efectos

## 🔧 Componentes Técnicos

### pose_detector.py
- Usa MediaPipe Pose (modelo lite)
- Detecta 33 puntos clave del cuerpo
- Enfocado en torso superior (landmarks 0-24)
- Suavizado temporal con buffer de 5 frames

### body_regions.py
- Calcula regiones útiles: hombros, torso, cabeza
- Genera contornos aproximados del cuerpo
- Proporciona coordenadas para posicionamiento de efectos
- Funciones de visualización para debug

### visual_effects.py
- 5 efectos visuales implementados
- Cada efecto como función independiente
- Usa técnicas de blending y overlays de OpenCV
- Colores y estilos diferenciados por efecto

### effect_renderer.py
- Gestiona transiciones suaves entre efectos
- Fade in/out automático (fade_speed configurable)
- Estabilización de intensidad
- Estado persistente del efecto activo

## 🎓 Valor Académico

Este proyecto demuestra:

1. **Integración de múltiples tecnologías de CV:**
   - Hand landmark detection
   - Pose estimation
   - Real-time rendering

2. **Arquitectura modular y escalable:**
   - Separación clara de responsabilidades
   - Componentes reutilizables
   - Fácil extensión con nuevos efectos

3. **Interacción humano-computador:**
   - Interfaz gestual natural
   - Respuesta visual inmersiva
   - Feedback en tiempo real

4. **Progresión técnica:**
   - Fase 1: Detección de gestos
   - Fase 2: Captura y entrenamiento de modelo
   - Fase 3: Clasificación con ML
   - **Fase 4: Interacción visual con transformaciones**

## 📊 Rendimiento

- **FPS esperado:** 25-30 fps en hardware moderno
- **Latencia:** < 50ms entre gesto y efecto
- **Resolución:** 640x480 (ajustable en main.py)

## 🔮 Posibles Extensiones

- [ ] Más efectos visuales (rayo, nieve, humo)
- [ ] Efectos combinados (múltiples gestos simultáneos)
- [ ] Efectos animados (partículas, movimiento)
- [ ] Segmentación avanzada del cuerpo
- [ ] Cambio de fondo (chroma key)
- [ ] Efectos de post-procesamiento (blur, stylization)

## 📝 Notas

- **Posición recomendada:** Sentado, torso superior visible
- **Iluminación:** Buena iluminación frontal para mejor detección
- **Distancia:** 1-2 metros de la cámara
- **Fondo:** Preferiblemente uniforme para mejor segmentación

---

**Proyecto:** Reconocimiento de Gestos con IA  
**Evolución:** Sistema de Interacción Visual en Tiempo Real  
**Tecnologías:** Python, OpenCV, MediaPipe, scikit-learn
