"""Golden Hour (Gesture 2) filter."""

import cv2
import numpy as np


def _apply_golden_hour(frame: np.ndarray) -> np.ndarray:
    result = frame.copy().astype(np.float32)

    result[:, :, 2] = np.clip(result[:, :, 2] * 1.15 + 20, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05 + 5, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.80 - 10, 0, 255)
    result = result.astype(np.uint8)

    result = cv2.convertScaleAbs(result, alpha=1.1, beta=8)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    h, w = result.shape[:2]
    y_grid, x_grid = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((x_grid - cx) / cx) ** 2 + ((y_grid - cy) / cy) ** 2)
    vignette = np.clip(1 - dist * 0.45, 0.55, 1.0)
    result = (result.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)

    return result


def apply_armor_overlay(frame: np.ndarray, body_regions, intensity: float = 1.0):
    filtered = _apply_golden_hour(frame)
    a = float(np.clip(intensity, 0.0, 1.0))
    frame[:] = cv2.addWeighted(filtered, a, frame, 1.0 - a, 0)
