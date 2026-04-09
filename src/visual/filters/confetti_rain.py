"""Confetti Rain (Gesture 4) filter."""

import random

import cv2
import numpy as np

_CONFETTI_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (128, 0, 255),
]
_CONFETTI_PARTICLES = []


def _create_confetti_particle(w: int, h: int, strong: bool = False) -> dict:
    speed = random.uniform(6, 14) if strong else random.uniform(3, 7)
    return {
        "x": random.uniform(0, w),
        "y": random.uniform(-30, 0),
        "vx": random.uniform(-2, 2),
        "vy": speed,
        "color": random.choice(_CONFETTI_COLORS),
        "width": random.randint(8, 18),
        "height": random.randint(4, 10),
        "angle": random.uniform(0, 360),
        "spin": random.uniform(-6, 6),
    }


def _hands_above_shoulders(body_regions) -> bool:
    key_points = getattr(body_regions, "key_points", {}) or {}
    needed = ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder"]
    if not all(k in key_points for k in needed):
        return False

    left_wrist = key_points["left_wrist"]
    right_wrist = key_points["right_wrist"]
    left_shoulder = key_points["left_shoulder"]
    right_shoulder = key_points["right_shoulder"]
    return left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]


def _update_and_draw_confetti(frame, strong: bool = False, intensity: float = 1.0):
    global _CONFETTI_PARTICLES
    h, w = frame.shape[:2]
    intensity = float(np.clip(intensity, 0.0, 1.0))

    base_new = 18 if strong else 6
    num_new = int(max(0, round(base_new * max(0.15, intensity))))
    for _ in range(num_new):
        _CONFETTI_PARTICLES.append(_create_confetti_particle(w, h, strong))

    if len(_CONFETTI_PARTICLES) > 400:
        _CONFETTI_PARTICLES = _CONFETTI_PARTICLES[-400:]

    overlay = frame.copy()
    alive = []
    for p in _CONFETTI_PARTICLES:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["angle"] += p["spin"]

        if p["y"] < h + 20:
            cx, cy = int(p["x"]), int(p["y"])
            angle_rad = np.deg2rad(p["angle"])
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            hw, hh = p["width"] // 2, p["height"] // 2

            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            corners = (corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(overlay, [corners], p["color"])
            alive.append(p)

    _CONFETTI_PARTICLES = alive
    frame[:] = cv2.addWeighted(overlay, intensity, frame, 1.0 - intensity, 0)


def apply_ice_shield(frame, body_regions, intensity: float = 1.0):
    strong = _hands_above_shoulders(body_regions)
    _update_and_draw_confetti(frame, strong=strong, intensity=intensity)
