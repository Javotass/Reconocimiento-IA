"""
dataset_collector.py
--------------------
Captura landmarks normalizados de la mano y los guarda en un CSV
para entrenar un modelo en la Fase 3.

Formato del CSV  (cabecera generada automáticamente)
──────────────────────────────────────────────────────
label, x0,y0,z0, x1,y1,z1, …, x20,y20,z20
  ↑ etiqueta del gesto   ↑ 21 landmarks × 3 coords = 63 columnas

Normalización aplicada
──────────────────────
  1. Traslación: se resta la muñeca (lm[0]) a todos los puntos.
  2. Escala: se divide por la distancia muñeca→nudillo_medio (lm[0]→lm[9]).
     Así el tamaño de la mano y la distancia a la cámara no afectan.

Uso desde main.py
─────────────────
  collector = DatasetCollector()
  collector.start_session("OK", duration_s=5)   # empieza a grabar "OK" 5 s

  # en cada frame del bucle:
  done, remaining = collector.tick(landmarks)   # landmarks: lista de 21 NormalizedLandmark

  # al salir:
  collector.save()
"""

import csv
import math
import os
import time
from typing import Optional

# ── constantes ────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DEFAULT_CSV   = os.path.join(DATASET_DIR, "dataset_landmarks.csv")
GESTURE_NAMES = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "OK"}
_HEADER       = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
# ─────────────────────────────────────────────────────────────────────────────


class DatasetCollector:
    """Graba muestras normalizadas de landmarks en un CSV."""

    def __init__(self, csv_path: Optional[str] = None):
        os.makedirs(DATASET_DIR, exist_ok=True)
        self._csv_path = csv_path or DEFAULT_CSV
        self._rows: list[list] = []

        # Estado de la sesión de grabación
        self._recording       = False
        self._session_label   = ""
        self._session_end_ts  = 0.0
        self._session_seconds = 0.0
        self._session_count   = 0       # muestras guardadas en esta sesión

        # Carga conteos previos si el archivo ya existe
        self._class_counts: dict[str, int] = self._load_counts()

    # ── API pública ───────────────────────────────────────────────────────────

    def start_session(self, label: str, duration_s: float = 5.0):
        """
        Empieza a grabar *duration_s* segundos bajo la etiqueta *label*.
        Llama a este método cuando el usuario pulsa la tecla de un gesto.
        """
        self._recording       = True
        self._session_label   = label
        self._session_seconds = duration_s
        self._session_end_ts  = time.perf_counter() + duration_s
        self._session_count   = 0
        print(f"[Dataset] Grabando '{label}' durante {duration_s:.0f}s …")

    def tick(self, landmarks) -> tuple[bool, float]:
        """
        Llamar en cada frame mientras haya landmarks disponibles.

        Parámetros
        ----------
        landmarks : lista de 21 NormalizedLandmark (lm.x, lm.y, lm.z)

        Devuelve
        --------
        session_done : bool   – True el frame en que termina la sesión
        remaining_s  : float  – segundos que quedan (0 si no hay sesión)
        """
        if not self._recording:
            return False, 0.0

        remaining = max(0.0, self._session_end_ts - time.perf_counter())

        if remaining <= 0.0:
            self._recording = False
            print(f"[Dataset] Sesión '{self._session_label}' finalizada "
                  f"({self._session_count} muestras).")
            self._class_counts[self._session_label] = (
                self._class_counts.get(self._session_label, 0) + self._session_count
            )
            return True, 0.0

        if landmarks is not None:
            norm = self._normalize(landmarks)
            if norm is not None:
                self._rows.append([self._session_label] + norm)
                self._session_count += 1

        return False, remaining

    def save(self):
        """Escribe todas las muestras acumuladas en el CSV (modo append)."""
        if not self._rows:
            print("[Dataset] No hay muestras nuevas que guardar.")
            return

        file_exists = os.path.isfile(self._csv_path)
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(_HEADER)
            writer.writerows(self._rows)

        total = sum(self._class_counts.values())
        print(f"[Dataset] Guardadas {len(self._rows)} muestras en {self._csv_path}")
        print(f"[Dataset] Total acumulado: {total} muestras")
        self._rows.clear()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_label(self) -> str:
        return self._session_label

    @property
    def class_counts(self) -> dict:
        return dict(self._class_counts)

    @property
    def pending_rows(self) -> int:
        return len(self._rows)

    # ── helpers privados ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(landmarks) -> Optional[list[float]]:
        """
        Normaliza los 21 landmarks:
          1. Traslación: resta lm[0] (muñeca)
          2. Escala: divide por dist(lm[0], lm[9])  (muñeca → nudillo medio)
        Devuelve una lista plana de 63 floats, o None si la escala es 0.
        """
        # 1. Traslación
        ox, oy, oz = landmarks[0].x, landmarks[0].y, landmarks[0].z
        pts = [(lm.x - ox, lm.y - oy, lm.z - oz) for lm in landmarks]

        # 2. Escala: distancia muñeca → nudillo dedo medio (índice 9)
        ref = math.sqrt(pts[9][0]**2 + pts[9][1]**2 + pts[9][2]**2)
        if ref < 1e-6:
            return None

        flat = [round(v / ref, 6) for pt in pts for v in pt]
        return flat

    def _load_counts(self) -> dict[str, int]:
        """Lee el CSV existente y cuenta muestras por clase."""
        counts: dict[str, int] = {}
        if not os.path.isfile(self._csv_path):
            return counts
        try:
            with open(self._csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)   # saltar cabecera
                for row in reader:
                    if row:
                        label = row[0]
                        counts[label] = counts.get(label, 0) + 1
        except Exception:
            pass
        return counts
