"""
pose_detector.py
----------------
Detects upper body pose using MediaPipe Pose (mediapipe >= 0.10.x).

Focused on seated position, tracking:
  - Head (nose, ears)
  - Shoulders (left, right)
  - Arms (elbows, wrists)
  - Upper torso

Temporal smoothing:
    Similar to gesture_detector, uses a rolling buffer to stabilize
    landmark positions and reduce jitter.

Key landmarks (from MediaPipe Pose 33-point model):
    0  = nose
    11 = left_shoulder
    12 = right_shoulder
    13 = left_elbow
    14 = right_elbow
    15 = left_wrist
    16 = right_wrist
    23 = left_hip
    24 = right_hip
"""

import collections
import os
import urllib.request
from typing import Optional
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── tuneable constants ────────────────────────────────────────────────────────
SMOOTHING_WINDOW = 5  # frames to average for position stability
MAX_LOST_FRAMES = 8
MIN_TRACKING_CONFIDENCE = 0.35

MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Key landmark indices for upper body
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Upper body connections for visualization
_UPPER_BODY_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13),  # left shoulder -> elbow
    (13, 15),  # left elbow -> wrist
    (12, 14),  # right shoulder -> elbow
    (14, 16),  # right elbow -> wrist
    (11, 23),  # left shoulder -> hip
    (12, 24),  # right shoulder -> hip
    (23, 24),  # hips
]
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _SimpleLandmark:
    """Lightweight landmark object used for smoothed landmark output."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


def _ensure_model() -> str:
    """Download the pose landmarker model if not present. Returns its path."""
    # Navegar a la raíz del proyecto (dos niveles arriba: src/core -> src -> raíz)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    if not os.path.isfile(model_path):
        print(f"[PoseDetector] Descargando modelo ({MODEL_FILENAME}) …")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[PoseDetector] Modelo guardado en {model_path}")
    return model_path


def _draw_pose_landmarks(frame: np.ndarray, landmarks, h: int, w: int):
    """Draw upper body landmarks and connections onto *frame* in-place."""
    pts = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]
    
    # Draw connections
    for a, b in _UPPER_BODY_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 200), 2, cv2.LINE_AA)
    
    # Draw key landmarks
    key_points = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, 
                  RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    for i in key_points:
        if i < len(pts):
            x, y = pts[i]
            cv2.circle(frame, (x, y), 5, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (30, 30, 30), 1, cv2.LINE_AA)


class PoseDetector:
    """Detects upper body pose from a BGR camera frame."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        max_lost_frames: int = MAX_LOST_FRAMES,
    ):
        model_path = _ensure_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        
        # Timestamp tracking for VIDEO mode
        self._frame_counter = 0
        
        # Smoothing buffer for landmark positions
        self._landmark_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

        # Tracking and confidence state
        self._min_tracking_confidence = min_tracking_confidence
        self._max_lost_frames = max_lost_frames
        self._last_valid_landmarks = None
        self._lost_frames = 0
        self._last_pose_confidence = 0.0
        self._tracking_state = "searching"  # searching | tracking | coasting | lost
        
        print("[PoseDetector] Inicializado correctamente")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> tuple[Optional[list], bool]:
        """
        Process a single BGR frame and return smoothed pose landmarks.

        Returns
        -------
        landmarks : list or None
            List of 33 NormalizedLandmark objects (x, y, z) if pose detected
        detected : bool
            True if a pose was successfully detected
        """
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_counter += 1
        timestamp_ms = self._frame_counter * 33  # ~30 fps

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return self._handle_pose_missing()

        # Get first person's pose (we only track one person)
        landmarks = result.pose_landmarks[0]
        
        confidence = self._compute_pose_confidence(landmarks)
        if confidence < self._min_tracking_confidence:
            return self._handle_pose_missing()

        # Valid pose: update tracking state and smooth.
        self._lost_frames = 0
        self._tracking_state = "tracking"
        self._last_pose_confidence = confidence

        # Add to smoothing buffer
        self._landmark_buffer.append(landmarks)

        # Return smoothed landmarks
        smoothed = self._smooth_landmarks()
        self._last_valid_landmarks = smoothed
        return smoothed, True

    def draw_on_frame(self, frame: np.ndarray, landmarks) -> None:
        """Draw pose visualization on the frame in-place."""
        if landmarks is None:
            return
        h, w = frame.shape[:2]
        _draw_pose_landmarks(frame, landmarks, h, w)

    def get_key_points(self, landmarks) -> dict:
        """
        Extract key upper body points as pixel coordinates.
        
        Returns a dict with keys: nose, left_shoulder, right_shoulder,
        left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip
        Each value is (x, y, z) in normalized coordinates [0, 1].
        """
        if landmarks is None:
            return {}
        
        points = {
            "nose": (landmarks[NOSE].x, landmarks[NOSE].y, landmarks[NOSE].z),
            "left_shoulder": (landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y, landmarks[LEFT_SHOULDER].z),
            "right_shoulder": (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y, landmarks[RIGHT_SHOULDER].z),
            "left_elbow": (landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y, landmarks[LEFT_ELBOW].z),
            "right_elbow": (landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y, landmarks[RIGHT_ELBOW].z),
            "left_wrist": (landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y, landmarks[LEFT_WRIST].z),
            "right_wrist": (landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y, landmarks[RIGHT_WRIST].z),
            "left_hip": (landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y, landmarks[LEFT_HIP].z),
            "right_hip": (landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y, landmarks[RIGHT_HIP].z),
        }
        return points

    def get_tracking_status(self) -> dict:
        """Expose pose tracking quality for debug HUD and tuning."""
        return {
            "state": self._tracking_state,
            "confidence": round(self._last_pose_confidence, 2),
            "lost_frames": self._lost_frames,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _smooth_landmarks(self):
        """Average landmark positions over the buffer window."""
        if not self._landmark_buffer:
            return None
        
        # Take the most recent landmarks as base
        base = self._landmark_buffer[-1]
        
        if len(self._landmark_buffer) < 2:
            return base
        
        # Create smoothed version by averaging positions.
        # Shoulders and hips get heavier smoothing than wrists to reduce jitter
        # while keeping arm motion responsive.
        smoothed = []
        num_landmarks = len(base)

        alpha_map = {
            LEFT_SHOULDER: 0.35,
            RIGHT_SHOULDER: 0.35,
            LEFT_HIP: 0.32,
            RIGHT_HIP: 0.32,
            LEFT_ELBOW: 0.28,
            RIGHT_ELBOW: 0.28,
            LEFT_WRIST: 0.22,
            RIGHT_WRIST: 0.22,
            NOSE: 0.30,
        }
        
        for i in range(num_landmarks):
            avg_x = sum(frame[i].x for frame in self._landmark_buffer) / len(self._landmark_buffer)
            avg_y = sum(frame[i].y for frame in self._landmark_buffer) / len(self._landmark_buffer)
            avg_z = sum(frame[i].z for frame in self._landmark_buffer) / len(self._landmark_buffer)

            prev = None
            if self._last_valid_landmarks is not None and i < len(self._last_valid_landmarks):
                prev = self._last_valid_landmarks[i]

            if prev is not None:
                alpha = alpha_map.get(i, 0.25)
                avg_x = prev.x * (1.0 - alpha) + avg_x * alpha
                avg_y = prev.y * (1.0 - alpha) + avg_y * alpha
                avg_z = prev.z * (1.0 - alpha) + avg_z * alpha

            vis = getattr(base[i], "visibility", 1.0)
            smoothed.append(_SimpleLandmark(avg_x, avg_y, avg_z, vis))
        
        return smoothed

    def _handle_pose_missing(self) -> tuple[Optional[list], bool]:
        """Keep the last stable pose for a few frames when detection is lost."""
        self._lost_frames += 1

        if self._last_valid_landmarks is not None and self._lost_frames <= self._max_lost_frames:
            self._tracking_state = "coasting"
            decay = 1.0 - (self._lost_frames / (self._max_lost_frames + 1))
            self._last_pose_confidence = max(0.0, self._last_pose_confidence * decay)
            return self._last_valid_landmarks, True

        self._tracking_state = "lost"
        self._last_pose_confidence = 0.0
        self._landmark_buffer.clear()
        self._last_valid_landmarks = None
        return None, False

    def _compute_pose_confidence(self, landmarks) -> float:
        """
        Heuristic confidence [0..1] based on:
          - key landmark visibility
          - torso size (too small usually means far/unstable)
          - proximity to frame borders
        """
        required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        if any(i >= len(landmarks) for i in required):
            return 0.0

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]

        shoulder_w = float(((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) ** 0.5)
        torso_h_l = float(((ls.x - lh.x) ** 2 + (ls.y - lh.y) ** 2) ** 0.5)
        torso_h_r = float(((rs.x - rh.x) ** 2 + (rs.y - rh.y) ** 2) ** 0.5)
        torso_h = 0.5 * (torso_h_l + torso_h_r)
        size_conf = min(1.0, (shoulder_w * torso_h) / 0.035)

        key = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_HIP, RIGHT_HIP]
        vis_vals = [float(getattr(landmarks[i], "visibility", 1.0)) for i in key if i < len(landmarks)]
        vis_conf = float(np.clip(np.mean(vis_vals), 0.0, 1.0)) if vis_vals else 0.0

        margin = 0.05
        border_penalty = 0.0
        for i in key:
            if i >= len(landmarks):
                continue
            x = landmarks[i].x
            y = landmarks[i].y
            overflow = max(margin - x, x - (1.0 - margin), margin - y, y - (1.0 - margin), 0.0)
            border_penalty = max(border_penalty, overflow)
        border_conf = max(0.0, 1.0 - border_penalty * 10.0)

        confidence = 0.50 * vis_conf + 0.35 * size_conf + 0.15 * border_conf
        return float(np.clip(confidence, 0.0, 1.0))

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_landmarker'):
            self._landmarker.close()
