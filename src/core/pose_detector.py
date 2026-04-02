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

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── tuneable constants ────────────────────────────────────────────────────────
SMOOTHING_WINDOW = 5  # frames to average for position stability

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
            return None, False

        # Get first person's pose (we only track one person)
        landmarks = result.pose_landmarks[0]
        
        # Add to smoothing buffer
        self._landmark_buffer.append(landmarks)
        
        # Return smoothed landmarks
        smoothed = self._smooth_landmarks()
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

    # ── private helpers ───────────────────────────────────────────────────────

    def _smooth_landmarks(self):
        """Average landmark positions over the buffer window."""
        if not self._landmark_buffer:
            return None
        
        # Take the most recent landmarks as base
        base = self._landmark_buffer[-1]
        
        if len(self._landmark_buffer) < 2:
            return base
        
        # Create smoothed version by averaging positions
        smoothed = []
        num_landmarks = len(base)
        
        for i in range(num_landmarks):
            avg_x = sum(frame[i].x for frame in self._landmark_buffer) / len(self._landmark_buffer)
            avg_y = sum(frame[i].y for frame in self._landmark_buffer) / len(self._landmark_buffer)
            avg_z = sum(frame[i].z for frame in self._landmark_buffer) / len(self._landmark_buffer)
            
            # Create a simple object with x, y, z attributes
            class SmoothLandmark:
                def __init__(self, x, y, z, visibility=1.0):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
            
            smoothed.append(SmoothLandmark(avg_x, avg_y, avg_z, base[i].visibility))
        
        return smoothed

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_landmarker'):
            self._landmarker.close()
