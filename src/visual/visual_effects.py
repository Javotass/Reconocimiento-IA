"""
visual_effects.py
-----------------
Visual effects to apply on the user's upper body based on detected gestures.

Each effect is a function that receives:
  - frame: the BGR image to modify
  - body_regions: BodyRegions object with calculated areas
  - intensity: float [0, 1] for effect strength

Effects available:
  1 (ONE)   → Shadow Mode       (dark transformation effect)
  2 (TWO)   → Armor Overlay     (metallic chest plate)
  3 (THREE) → Fire Shoulders    (flames on shoulders)
  4 (FOUR)  → Ice Shield        (frozen effect on torso)
  5 (OK)    → Golden Crown      (enhanced crown above head)
"""

import cv2
import numpy as np
import math


# ── Helper Functions ──────────────────────────────────────────────────────────

def _create_soft_mask(shape, contour, blur_ksize=31):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return mask


def _apply_local_color_overlay(frame, mask, color_bgr, alpha):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = color_bgr

    mask_f = (mask.astype(np.float32) / 255.0) * alpha
    mask_f = mask_f[..., None]

    frame[:] = (frame.astype(np.float32) * (1.0 - mask_f) +
                overlay.astype(np.float32) * mask_f).astype(np.uint8)


def _draw_glow_outline(frame, contour, color, thickness=10, blur_size=21, alpha=0.5):
    glow = np.zeros_like(frame, dtype=np.uint8)
    cv2.polylines(glow, [contour.astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (blur_size, blur_size), 0)
    cv2.addWeighted(glow, alpha, frame, 1.0, 0, frame)


def _desaturate_frame(frame, amount=0.35):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1.0 - amount)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ── Effect 1: Shadow Mode ─────────────────────────────────────────────────────

def apply_shadow_mode(frame: np.ndarray, body_regions, intensity: float = 1.0):
    """
    Apply dark transformation effect.
    Darkens upper body, desaturates background, adds soft outline.
    """
    contour = body_regions.get_upper_body_contour()
    if contour is None:
        return

    # ampliar un poco la silueta para que cubra mejor cabeza/torso
    center = contour.mean(axis=0)
    expanded = []
    for pt in contour:
        v = pt - center
        expanded.append(center + v * 1.08)
    expanded = np.array(expanded, dtype=np.int32)

    # 1) desaturar ligeramente el frame entero
    base = _desaturate_frame(frame.copy(), amount=0.25 * intensity)

    # 2) máscara suave del cuerpo
    mask = _create_soft_mask(frame.shape, expanded, blur_ksize=41)

    # 3) oscurecer cuerpo con tinte negro/azulado
    shadow_layer = base.copy()
    _apply_local_color_overlay(
        shadow_layer,
        mask,
        color_bgr=(20, 10, 10),   # negro con toque frío
        alpha=0.78 * intensity
    )

    # 4) mezclar resultado
    frame[:] = shadow_layer

    # 5) borde glow sutil
    _draw_glow_outline(
        frame,
        expanded,
        color=(120, 180, 255),
        thickness=max(4, int(8 * intensity)),
        blur_size=31,
        alpha=0.20 + 0.20 * intensity
    )

    # 6) borde interno oscuro para marcar silueta
    cv2.polylines(
        frame,
        [expanded],
        True,
        (40, 40, 40),
        max(2, int(3 * intensity)),
        cv2.LINE_AA
    )


# ── Effect 2: Armor Overlay ───────────────────────────────────────────────────

def apply_armor_overlay(frame: np.ndarray, body_regions, intensity: float = 1.0):
    """
    Draw a metallic armor plate over the torso.
    Color: silver/steel gray with metallic sheen.
    """
    torso = body_regions.get_torso_region()
    if torso is None:
        return

    overlay = frame.copy()
    
    # Define armor plate polygon (slightly smaller than torso)
    tl = torso["top_left"]
    tr = torso["top_right"]
    bl = torso["bottom_left"]
    br = torso["bottom_right"]
    
    # Shrink inward by 10%
    center = torso["center"]
    def shrink_point(pt, center, factor=0.9):
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        return (int(center[0] + dx * factor), int(center[1] + dy * factor))
    
    armor_pts = np.array([
        shrink_point(tl, center),
        shrink_point(tr, center),
        shrink_point(br, center),
        shrink_point(bl, center),
    ], dtype=np.int32)
    
    # Draw metallic plate
    armor_color = (140, 140, 160)  # steel gray
    cv2.fillPoly(overlay, [armor_pts], armor_color, lineType=cv2.LINE_AA)
    
    # Add metallic highlights (lighter stripes)
    highlight_color = (200, 200, 220)
    mid_y = (armor_pts[0][1] + armor_pts[2][1]) // 2
    cv2.line(overlay, 
             (armor_pts[0][0], mid_y), 
             (armor_pts[1][0], mid_y),
             highlight_color, 3, cv2.LINE_AA)
    
    # Add border/edges
    cv2.polylines(overlay, [armor_pts], isClosed=True,
                 color=(80, 80, 100), thickness=3, lineType=cv2.LINE_AA)
    
    # Blend with transparency
    alpha = 0.6 * intensity
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Effect 3: Fire Shoulders ──────────────────────────────────────────────────

def apply_fire_shoulders(frame: np.ndarray, body_regions, intensity: float = 1.0):
    """
    Draw flame effects on both shoulders.
    Color: orange/red with animated particles.
    """
    shoulder = body_regions.get_shoulder_region()
    if shoulder is None:
        return

    overlay = frame.copy()
    
    left_pos = shoulder["left"]
    right_pos = shoulder["right"]
    
    # Draw flame effect on each shoulder
    for pos in [left_pos, right_pos]:
        # Multiple flame layers
        flame_colors = [
            (0, 50, 255),   # bright red-orange core
            (0, 120, 255),  # orange
            (0, 180, 200),  # yellow-orange outer
        ]
        
        radii = [15, 25, 35]
        
        for color, radius in zip(flame_colors, radii):
            # Draw with decreasing opacity
            alpha = int(200 * intensity * (1 - radius / 50))
            cv2.circle(overlay, pos, int(radius * intensity), color, -1, cv2.LINE_AA)
    
    # Blend
    cv2.addWeighted(overlay, 0.5 * intensity, frame, 1 - 0.25 * intensity, 0, frame)


# ── Effect 4: Ice Shield ──────────────────────────────────────────────────────

def apply_ice_shield(frame: np.ndarray, body_regions, intensity: float = 1.0):
    """
    Draw an ice/frozen shield effect on the torso.
    Color: light blue/cyan with crystalline patterns.
    """
    torso = body_regions.get_torso_region()
    if torso is None:
        return

    overlay = frame.copy()
    
    # Define shield polygon
    tl = torso["top_left"]
    tr = torso["top_right"]
    bl = torso["bottom_left"]
    br = torso["bottom_right"]
    center = torso["center"]
    
    # Draw semi-transparent ice plate
    shield_pts = np.array([tl, tr, br, bl], dtype=np.int32)
    ice_color = (255, 200, 150)  # light cyan
    cv2.fillPoly(overlay, [shield_pts], ice_color, lineType=cv2.LINE_AA)
    
    # Add crystalline pattern (diagonal lines)
    crystal_color = (255, 255, 200)  # brighter cyan
    num_lines = 6
    for i in range(num_lines):
        t = i / num_lines
        # Diagonal lines from top-left to bottom-right
        x1 = int(tl[0] + (tr[0] - tl[0]) * t)
        y1 = tl[1]
        x2 = int(bl[0] + (br[0] - bl[0]) * t)
        y2 = bl[1]
        cv2.line(overlay, (x1, y1), (x2, y2), crystal_color, 1, cv2.LINE_AA)
    
    # Border glow
    cv2.polylines(overlay, [shield_pts], isClosed=True,
                 color=(255, 255, 180), thickness=3, lineType=cv2.LINE_AA)
    
    # Blend
    alpha = 0.4 * intensity
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Effect 5: Golden Crown ────────────────────────────────────────────────────

def apply_golden_crown(frame: np.ndarray, body_regions, intensity: float = 1.0):
    """
    Draw an enhanced golden crown above the head.
    Features: better proportions, shadow, glow, gems.
    """
    head = body_regions.get_head_region()
    if head is None:
        return

    center = head["center"]
    radius = max(20, head["radius"])

    crown_w = int(radius * 2.0)
    crown_h = int(radius * 1.0)

    cx = center[0]
    cy = center[1] - int(radius * 1.35)

    overlay = np.zeros_like(frame, dtype=np.uint8)

    left = cx - crown_w // 2
    right = cx + crown_w // 2
    top = cy - crown_h // 2
    bottom = cy + crown_h // 2

    # base curva
    base_pts = np.array([
        (left, bottom - crown_h // 5),
        (left + crown_w // 8, bottom),
        (right - crown_w // 8, bottom),
        (right, bottom - crown_h // 5),
        (right - crown_w // 12, bottom - crown_h // 2),
        (left + crown_w // 12, bottom - crown_h // 2),
    ], dtype=np.int32)

    # puntas principales
    p1 = (left + crown_w // 6, top + crown_h // 3)
    p2 = (cx - crown_w // 6, top)
    p3 = (cx, top + crown_h // 5)
    p4 = (cx + crown_w // 6, top)
    p5 = (right - crown_w // 6, top + crown_h // 3)

    crown_pts = np.array([
        (left + crown_w // 12, bottom - crown_h // 2),
        p1,
        p2,
        p3,
        p4,
        p5,
        (right - crown_w // 12, bottom - crown_h // 2),
        (right - crown_w // 8, bottom),
        (left + crown_w // 8, bottom),
    ], dtype=np.int32)

    gold_main = (0, 195, 255)
    gold_dark = (0, 120, 190)
    gold_light = (80, 235, 255)

    # sombra
    shadow = np.zeros_like(frame, dtype=np.uint8)
    shadow_pts = crown_pts.copy()
    shadow_pts[:, 1] += 6
    cv2.fillPoly(shadow, [shadow_pts], (0, 60, 90), lineType=cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    cv2.addWeighted(shadow, 0.35 * intensity, frame, 1.0, 0, frame)

    # cuerpo corona
    cv2.fillPoly(overlay, [crown_pts], gold_main, lineType=cv2.LINE_AA)

    # brillo superior
    highlight_pts = crown_pts.copy()
    highlight_pts[:, 1] = np.maximum(highlight_pts[:, 1] - 4, 0)
    cv2.polylines(overlay, [highlight_pts], True, gold_light, 2, cv2.LINE_AA)

    # base
    cv2.fillPoly(overlay, [base_pts], (0, 170, 235), lineType=cv2.LINE_AA)

    # borde
    cv2.polylines(overlay, [crown_pts], True, gold_dark, 3, cv2.LINE_AA)

    # gemas
    gem_positions = [
        (cx - crown_w // 4, top + crown_h // 3),
        (cx, top + crown_h // 4),
        (cx + crown_w // 4, top + crown_h // 3),
    ]
    gem_colors = [(255, 80, 80), (255, 220, 80), (180, 80, 255)]

    for pos, color in zip(gem_positions, gem_colors):
        cv2.circle(overlay, pos, max(4, radius // 7), color, -1, cv2.LINE_AA)
        cv2.circle(overlay, pos, max(4, radius // 7), (255, 255, 255), 1, cv2.LINE_AA)

    # glow exterior
    glow = np.zeros_like(frame, dtype=np.uint8)
    cv2.polylines(glow, [crown_pts], True, gold_light, 8, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (21, 21), 0)
    cv2.addWeighted(glow, 0.28 * intensity, frame, 1.0, 0, frame)

    # composición final
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    alpha = ((mask.astype(np.float32) / 255.0) * (0.85 * intensity))[..., None]

    frame[:] = (frame.astype(np.float32) * (1.0 - alpha) +
                overlay.astype(np.float32) * alpha).astype(np.uint8)


# ── Main Effect Dispatcher ────────────────────────────────────────────────────

# Mapping gesture → effect function
EFFECT_MAP = {
    1: apply_shadow_mode,
    2: apply_armor_overlay,
    3: apply_fire_shoulders,
    4: apply_ice_shield,
    5: apply_golden_crown,
}

EFFECT_NAMES = {
    1: "Shadow Mode",
    2: "Armor Overlay",
    3: "Fire Shoulders",
    4: "Ice Shield",
    5: "Golden Crown",
}


def apply_effect_for_gesture(
    frame: np.ndarray,
    gesture: int,
    body_regions,
    intensity: float = 1.0
):
    """
    Apply the visual effect corresponding to the detected gesture.
    
    Parameters
    ----------
    frame : np.ndarray
        BGR image to modify in-place
    gesture : int
        Detected gesture (1-5)
    body_regions : BodyRegions
        Calculated body regions
    intensity : float
        Effect intensity [0, 1]
    """
    effect_func = EFFECT_MAP.get(gesture)
    if effect_func is not None:
        effect_func(frame, body_regions, intensity)


def get_effect_name(gesture: int) -> str:
    """Get the name of the effect for a given gesture."""
    return EFFECT_NAMES.get(gesture, "No Effect")
