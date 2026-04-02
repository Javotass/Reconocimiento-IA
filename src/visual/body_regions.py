"""
body_regions.py
---------------
Calculates useful body regions from pose landmarks for visual effect placement.

Regions calculated:
  - Shoulder area (bounding box between shoulders)
  - Torso area (from shoulders to hips)
  - Head area (around nose position)
  - Body contour (approximate polygon)

All coordinates are returned in normalized [0, 1] format and can be
converted to pixel coordinates by multiplying by frame dimensions.
"""

import math
from typing import Optional

import cv2
import numpy as np


class BodyRegions:
    """Calculates and stores useful body regions from pose landmarks."""

    def __init__(self):
        self.landmarks = None
        self.key_points = {}
        self.frame_width = 640
        self.frame_height = 480

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, landmarks, key_points: dict, frame_width: int, frame_height: int):
        """Update regions based on new landmarks."""
        self.landmarks = landmarks
        self.key_points = key_points
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_shoulder_region(self) -> Optional[dict]:
        """
        Get shoulder region as a bounding box.
        
        Returns dict with keys: center, width, height, angle (all in pixels).
        """
        if "left_shoulder" not in self.key_points or "right_shoulder" not in self.key_points:
            return None

        ls = self.key_points["left_shoulder"]
        rs = self.key_points["right_shoulder"]

        # Convert to pixel coordinates
        ls_px = (int(ls[0] * self.frame_width), int(ls[1] * self.frame_height))
        rs_px = (int(rs[0] * self.frame_width), int(rs[1] * self.frame_height))

        # Calculate center, width, and angle
        center_x = (ls_px[0] + rs_px[0]) // 2
        center_y = (ls_px[1] + rs_px[1]) // 2
        width = int(math.dist(ls_px, rs_px))
        
        # Angle of shoulder line (for rotation)
        dx = rs_px[0] - ls_px[0]
        dy = rs_px[1] - ls_px[1]
        angle = math.degrees(math.atan2(dy, dx))

        # Height based on shoulder width (approximate)
        height = int(width * 0.6)

        return {
            "center": (center_x, center_y),
            "width": width,
            "height": height,
            "angle": angle,
            "left": ls_px,
            "right": rs_px,
        }

    def get_torso_region(self) -> Optional[dict]:
        """
        Get torso region from shoulders to hips.
        
        Returns dict with keys: top_left, top_right, bottom_left, bottom_right,
        center, width, height (all in pixels).
        """
        if not all(k in self.key_points for k in 
                   ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            return None

        ls = self._to_pixels(self.key_points["left_shoulder"])
        rs = self._to_pixels(self.key_points["right_shoulder"])
        lh = self._to_pixels(self.key_points["left_hip"])
        rh = self._to_pixels(self.key_points["right_hip"])

        # Calculate bounding region
        top_left = ls
        top_right = rs
        bottom_left = lh
        bottom_right = rh

        center_x = (ls[0] + rs[0] + lh[0] + rh[0]) // 4
        center_y = (ls[1] + rs[1] + lh[1] + rh[1]) // 4

        width = int((math.dist(ls, rs) + math.dist(lh, rh)) / 2)
        height = int((math.dist(ls, lh) + math.dist(rs, rh)) / 2)

        return {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "center": (center_x, center_y),
            "width": width,
            "height": height,
        }

    def get_head_region(self) -> Optional[dict]:
        """
        Get head region centered on nose.
        
        Returns dict with keys: center, radius (estimated head size).
        """
        if "nose" not in self.key_points or "left_shoulder" not in self.key_points:
            return None

        nose = self._to_pixels(self.key_points["nose"])
        ls = self._to_pixels(self.key_points["left_shoulder"])
        rs = self._to_pixels(self.key_points["right_shoulder"])

        # Estimate head radius based on shoulder width
        shoulder_width = math.dist(ls, rs)
        head_radius = int(shoulder_width * 0.35)  # head is ~35% of shoulder width

        return {
            "center": nose,
            "radius": head_radius,
        }

    def get_upper_body_contour(self) -> Optional[np.ndarray]:
        """
        Get approximate contour polygon for upper body.
        
        Returns numpy array of points [(x1,y1), (x2,y2), ...] suitable for
        cv2.fillPoly or cv2.polylines.
        """
        shoulder_reg = self.get_shoulder_region()
        torso_reg = self.get_torso_region()
        head_reg = self.get_head_region()

        if not all([shoulder_reg, torso_reg, head_reg]):
            return None

        # Build contour from key points
        points = []
        
        # Start from top (head)
        head_top = (head_reg["center"][0], head_reg["center"][1] - head_reg["radius"])
        points.append(head_top)

        # Go around shoulders
        points.append(shoulder_reg["left"])
        points.append(torso_reg["bottom_left"])
        points.append(torso_reg["bottom_right"])
        points.append(shoulder_reg["right"])

        # Close at head
        points.append(head_top)

        return np.array(points, dtype=np.int32)

    def get_body_bounds(self) -> Optional[dict]:
        """
        Get overall bounding box for the upper body.
        
        Returns dict with: x, y, width, height (in pixels).
        """
        if "nose" not in self.key_points:
            return None

        all_points = []
        for key in ["nose", "left_shoulder", "right_shoulder", 
                    "left_elbow", "right_elbow", "left_wrist", 
                    "right_wrist", "left_hip", "right_hip"]:
            if key in self.key_points:
                px = self._to_pixels(self.key_points[key])
                all_points.append(px)

        if not all_points:
            return None

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add some padding
        padding = 20
        return {
            "x": max(0, x_min - padding),
            "y": max(0, y_min - padding),
            "width": min(self.frame_width, x_max - x_min + 2 * padding),
            "height": min(self.frame_height, y_max - y_min + 2 * padding),
        }

    # ── visualization helpers ──────────────────────────────────────────────────

    def draw_shoulder_region(self, frame: np.ndarray, color=(0, 255, 200), thickness=2):
        """Draw shoulder region on frame for debugging."""
        region = self.get_shoulder_region()
        if region is None:
            return

        # Draw line between shoulders
        cv2.line(frame, region["left"], region["right"], color, thickness, cv2.LINE_AA)
        
        # Draw center point
        cv2.circle(frame, region["center"], 5, color, -1, cv2.LINE_AA)

    def draw_torso_region(self, frame: np.ndarray, color=(255, 200, 0), thickness=2):
        """Draw torso quad on frame for debugging."""
        region = self.get_torso_region()
        if region is None:
            return

        points = np.array([
            region["top_left"],
            region["top_right"],
            region["bottom_right"],
            region["bottom_left"],
        ], dtype=np.int32)

        cv2.polylines(frame, [points], isClosed=True, color=color, 
                     thickness=thickness, lineType=cv2.LINE_AA)

    def draw_head_region(self, frame: np.ndarray, color=(200, 100, 255), thickness=2):
        """Draw head circle on frame for debugging."""
        region = self.get_head_region()
        if region is None:
            return

        cv2.circle(frame, region["center"], region["radius"], color, 
                  thickness, cv2.LINE_AA)

    def draw_all_regions(self, frame: np.ndarray):
        """Draw all regions for debugging."""
        self.draw_shoulder_region(frame, color=(0, 255, 200))
        self.draw_torso_region(frame, color=(255, 200, 0))
        self.draw_head_region(frame, color=(200, 100, 255))

    # ── private helpers ───────────────────────────────────────────────────────

    def _to_pixels(self, normalized_point: tuple) -> tuple:
        """Convert normalized (x, y, z) to pixel coordinates (x, y)."""
        x = int(normalized_point[0] * self.frame_width)
        y = int(normalized_point[1] * self.frame_height)
        return (x, y)
