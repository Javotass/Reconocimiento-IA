"""
gesture_classifier.py
---------------------
Carga el modelo entrenado y proporciona una interfaz simple
para predecir gestos a partir de landmarks normalizados.

Se usa DENTRO de gesture_detector.py como sustituto de las reglas geométricas.
Si el modelo no existe, gesture_detector.py sigue usando las reglas (fallback).

Uso directo
───────────
    clf = GestureClassifier()
    if clf.available:
        gesture_int = clf.predict(landmarks)   # landmarks: lista de 21 NormalizedLandmark
"""

import math
import os
from typing import Optional

import joblib
import numpy as np

# ── rutas ─────────────────────────────────────────────────────────────────────
# Navegar a la raíz del proyecto (dos niveles arriba: src/core -> src -> raíz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
MODEL_PATH   = os.path.join(DATASET_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(DATASET_DIR, "label_encoder.pkl")

# Mapeo etiqueta string → entero (mismo que en el resto del sistema)
_LABEL_TO_INT = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "OK": 5}


class GestureClassifier:
    """
    Wrapper sobre el modelo sklearn entrenado.

    Atributos
    ─────────
    available : bool  – True si el modelo está cargado y listo
    model_name: str   – nombre del tipo de modelo cargado
    """

    def __init__(self):
        self._model   = None
        self._encoder = None
        self._load()

    # ── API pública ───────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._model is not None and self._encoder is not None

    @property
    def model_name(self) -> str:
        if self._model is None:
            return "sin modelo"
        return type(self._model).__name__

    def predict(self, landmarks) -> int:
        """
        Predice el gesto a partir de 21 NormalizedLandmark.

        Devuelve un entero: 1/2/3/4/5  ó  0 si no hay confianza suficiente.
        Usa la misma normalización que DatasetCollector._normalize().
        """
        if not self.available:
            return 0

        features = self._normalize(landmarks)
        if features is None:
            return 0

        X         = np.array(features, dtype=np.float32).reshape(1, -1)
        probs     = self._model.predict_proba(X)[0]
        max_prob  = float(probs.max())
        pred_idx  = int(probs.argmax())

        # umbral de confianza mínima
        if max_prob < 0.55:
            return 0

        label = self._encoder.classes_[pred_idx]
        return _LABEL_TO_INT.get(label, 0)

    def predict_with_confidence(self, landmarks) -> tuple[int, float, str]:
        """
        Versión extendida que devuelve (gesto, confianza, etiqueta_str).
        Útil para el panel de debug.
        """
        if not self.available:
            return 0, 0.0, "?"

        features = self._normalize(landmarks)
        if features is None:
            return 0, 0.0, "?"

        X        = np.array(features, dtype=np.float32).reshape(1, -1)
        probs    = self._model.predict_proba(X)[0]
        max_prob = float(probs.max())
        pred_idx = int(probs.argmax())
        label    = self._encoder.classes_[pred_idx]

        gesture = _LABEL_TO_INT.get(label, 0) if max_prob >= 0.55 else 0
        return gesture, round(max_prob, 3), label

    # ── helpers privados ──────────────────────────────────────────────────────

    def _load(self):
        """Intenta cargar modelo y encoder desde disco."""
        if os.path.isfile(MODEL_PATH) and os.path.isfile(ENCODER_PATH):
            try:
                self._model   = joblib.load(MODEL_PATH)
                self._encoder = joblib.load(ENCODER_PATH)
                print(f"[GestureClassifier] Modelo cargado: {self.model_name}")
            except Exception as e:
                print(f"[GestureClassifier] Error al cargar modelo: {e}")
                self._model = self._encoder = None
        else:
            print("[GestureClassifier] Modelo no encontrado "
                  "→ usando clasificador por reglas.")

    @staticmethod
    def _normalize(landmarks) -> Optional[list]:
        """
        Misma normalización que DatasetCollector:
          1. Traslación: resta lm[0] (muñeca)
          2. Escala: divide por dist(lm[0], lm[9])
        Devuelve lista de 63 floats o None.
        """
        ox, oy, oz = landmarks[0].x, landmarks[0].y, landmarks[0].z
        pts = [(lm.x - ox, lm.y - oy, lm.z - oz) for lm in landmarks]

        ref = math.sqrt(pts[9][0]**2 + pts[9][1]**2 + pts[9][2]**2)
        if ref < 1e-6:
            return None

        return [round(v / ref, 6) for pt in pts for v in pt]
