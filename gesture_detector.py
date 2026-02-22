"""
gesture_detector.py
-------------------
Detects hand gestures (1, 2 or 3 extended fingers) using the
MediaPipe Tasks API (mediapipe >= 0.10.x).

The old mp.solutions.hands was removed in mediapipe 0.10.21+;
this module uses mediapipe.tasks.python.vision.HandLandmarker instead.

Temporal smoothing
    A rolling buffer of the last BUFFER_SIZE gesture readings is kept.
    The gesture with the highest count (majority vote) is returned, but only
    if it surpasses MIN_CONFIDENCE fraction of the buffer.
"""

import collections
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── tuneable constants ────────────────────────────────────────────────────────
BUFFER_SIZE    = 10
MIN_CONFIDENCE = 0.5

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmark indices
_TIP = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_PIP = [3, 6, 10, 14, 18]   # corresponding PIP / IP joints

# Hand connections for manual rendering (standard 21-point topology)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # outer palm
]
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_model() -> str:
    """Download the hand landmarker model if not present. Returns its path."""
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILENAME)
    if not os.path.isfile(model_path):
        print(f"[GestureDetector] Descargando modelo ({MODEL_FILENAME}) …")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[GestureDetector] Modelo guardado en {model_path}")
    return model_path


def _draw_landmarks(frame: np.ndarray, landmarks, h: int, w: int):
    """Draw 21 hand landmarks and connections onto *frame* in-place."""
    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        radius = 7 if i in _TIP else 4
        color  = (0, 255, 140) if i in _TIP else (255, 255, 255)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (30, 30, 30), 1, cv2.LINE_AA)


class GestureDetector:
    """Detects 1 / 2 / 3 finger gestures from a BGR camera frame."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float  = 0.6,
    ):
        model_path = _ensure_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector   = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms: int = 0

        self._buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._last_confirmed: int = 0
        self._last_hand_side: str = "?"
        self._last_confidence: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray):
        """
        Run detection on *frame_bgr* (BGR uint8 ndarray).

        Returns
        -------
        gesture    : int   – smoothed gesture (0 = none, 1 / 2 / 3 = fingers)
        annotated  : ndarray – frame with landmarks drawn
        raw_gesture: int   – instantaneous (unsmoothed) gesture this frame
        """
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33          # ~30 fps tick; must be monotonic
        results = self._detector.detect_for_video(mp_image, self._timestamp_ms)

        raw_gesture = 0
        confidence  = 0.0

        if results.hand_landmarks and results.handedness:
            for landmarks, handedness in zip(
                results.hand_landmarks, results.handedness
            ):
                self._last_hand_side = handedness[0].category_name  # "Left"/"Right"
                confidence = self._compute_confidence(landmarks, h, w)

                if confidence >= 0.35:   # ignora mano muy pequeña o cortada
                    _draw_landmarks(annotated, landmarks, h, w)
                    raw_gesture = self._count_fingers(landmarks, self._last_hand_side)
                break   # primera mano únicamente

        self._last_confidence = confidence
        self._buffer.append(raw_gesture)
        smoothed = self._majority_vote()
        return smoothed, annotated, raw_gesture, confidence, self._last_hand_side

    def reset(self):
        """Clear the smoothing buffer."""
        self._buffer.clear()
        self._last_confirmed = 0

    def __del__(self):
        try:
            self._detector.close()
        except Exception:
            pass

    # ── private helpers ───────────────────────────────────────────────────────

    def _count_fingers(self, landmarks, handedness: str) -> int:
        """
        Count extended non-thumb fingers and return gesture id (0–4).

        Non-thumb: tip.y < pip.y  →  finger is up
        Thumb:  "Right" hand → tip.x < ip.x  |  "Left" hand → tip.x > ip.x
        OK gesture (4): thumb tip near index tip + middle/ring/pinky extended
        """
        lm = landmarks

        # ── OK gesture (5): pulgar e índice se tocan, resto extendidos ───────
        thumb_index_dist = (
            (lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2
        ) ** 0.5
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y
        if thumb_index_dist < 0.08 and middle_up and ring_up and pinky_up:
            return 5

        # ── gestos 1 / 2 / 3 / 4 (dedos no-pulgar extendidos) ───────────────
        if handedness == "Right":
            thumb_up = lm[_TIP[0]].x < lm[_PIP[0]].x
        else:
            thumb_up = lm[_TIP[0]].x > lm[_PIP[0]].x

        fingers_up = sum(
            lm[_TIP[i]].y < lm[_PIP[i]].y
            for i in range(1, 5)
        )

        if fingers_up == 1:
            return 1
        elif fingers_up == 2:
            return 2
        elif fingers_up == 3:
            return 3
        elif fingers_up == 4:
            return 4
        else:
            return 0

    @staticmethod
    def _compute_confidence(landmarks, h: int, w: int) -> float:
        """
        Heuristic confidence [0..1] based on:
          - hand visible size (too small = far away)
          - proximity to frame border (landmarks cut off)
        """
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Size confidence: bbox area relative to frame
        bbox_area  = (max_x - min_x) * (max_y - min_y)
        size_conf  = min(1.0, bbox_area / 0.045)   # ~21% de área = confianza plena

        # Border confidence: penaliza si algún landmark sale del margen
        margin = 0.07
        overflow = max(
            margin - min_x,
            max_x - (1.0 - margin),
            margin - min_y,
            max_y - (1.0 - margin),
            0.0,
        )
        border_conf = max(0.0, 1.0 - overflow * 12)

        return round(size_conf * border_conf, 2)

    def _majority_vote(self) -> int:
        """Return the most common value in the buffer if it meets MIN_CONFIDENCE."""
        if not self._buffer:
            return self._last_confirmed

        counts = collections.Counter(self._buffer)
        most_common_gesture, most_common_count = counts.most_common(1)[0]

        if most_common_count / len(self._buffer) >= MIN_CONFIDENCE:
            self._last_confirmed = most_common_gesture

        return self._last_confirmed
