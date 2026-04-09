"""Fire Shoulders (Gesture 3) filter."""

import math
import time

import cv2


def apply_fire_shoulders(frame, body_regions, intensity: float = 1.0):
    shoulder = body_regions.get_shoulder_region()
    if shoulder is None:
        return

    overlay = frame.copy()
    left_pos = shoulder["left"]
    right_pos = shoulder["right"]

    t = time.perf_counter()
    for pos in [left_pos, right_pos]:
        jitter_x = int(4 * math.sin(t * 11.0 + pos[0] * 0.03))
        jitter_y = int(6 * math.cos(t * 9.0 + pos[1] * 0.04))
        p = (pos[0] + jitter_x, pos[1] + jitter_y)

        flame_colors = [
            (10, 40, 255),
            (0, 115, 255),
            (0, 180, 255),
            (80, 235, 255),
        ]
        radii = [10, 18, 28, 42]

        for i, (color, radius) in enumerate(zip(flame_colors, radii)):
            rr = max(1, int(radius * (0.82 + 0.22 * math.sin(t * (7 + i) + i))))
            cv2.circle(overlay, p, rr, color, -1, cv2.LINE_AA)

        for k in range(8):
            ang = (k / 8.0) * 2 * math.pi + t * 1.7
            px = int(p[0] + math.cos(ang) * (18 + 9 * math.sin(t + k)))
            py = int(p[1] - 20 + math.sin(ang) * 10)
            cv2.circle(overlay, (px, py), 2, (120, 240, 255), -1, cv2.LINE_AA)

    cv2.addWeighted(cv2.GaussianBlur(overlay, (25, 25), 0), 0.30 * intensity, frame, 1.0, 0, frame)
    cv2.addWeighted(overlay, 0.58 * intensity, frame, 1.0 - 0.22 * intensity, 0, frame)
