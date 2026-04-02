"""
src.core
--------
Módulos de detección y clasificación.

Exports:
  - GestureDetector: Detector de gestos de mano
  - PoseDetector: Detector de pose del torso superior
  - GestureClassifier: Clasificador ML de gestos
"""

from .gesture_detector import GestureDetector
from .pose_detector import PoseDetector

try:
    from .gesture_classifier import GestureClassifier
except ImportError:
    GestureClassifier = None

__all__ = ["GestureDetector", "PoseDetector", "GestureClassifier"]
