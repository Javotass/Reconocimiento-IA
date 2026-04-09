"""Shadow mode (Gesture 1): Netherite chestplate overlay."""

import math
import os

import cv2
import numpy as np

_NETHERITE_TEXTURE = None
_NETHERITE_TEXTURE_LOADED = False
_NETHERITE_TEXTURE_WARNED = False
_MIRROR_CHESTPLATE_TEXTURE = True
_FLIP_CHESTPLATE_VERTICAL = True


def _project_point(pt, w: int, h: int):
    return (int(pt[0] * w), int(pt[1] * h))


def _get_netherite_texture() -> np.ndarray | None:
    global _NETHERITE_TEXTURE, _NETHERITE_TEXTURE_LOADED
    if _NETHERITE_TEXTURE_LOADED:
        return _NETHERITE_TEXTURE

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    candidates = [
        os.path.join(base_dir, "images", "netherite_chestplate.png"),
        os.path.join(base_dir, "images", "chestplate_netherite.png"),
        os.path.join(base_dir, "images", "netherite.png"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            tex = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if tex is not None and tex.size > 0:
                _NETHERITE_TEXTURE = tex
                break

    _NETHERITE_TEXTURE_LOADED = True
    return _NETHERITE_TEXTURE


def _overlay_rgba_rotated(
    frame: np.ndarray,
    rgba: np.ndarray,
    center: tuple[int, int],
    angle_deg: float,
    alpha_scale: float = 1.0,
):
    h, w = rgba.shape[:2]
    if h < 2 or w < 2:
        return

    rot_m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        rgba,
        rot_m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    x0 = int(center[0] - w / 2)
    y0 = int(center[1] - h / 2)
    x1 = x0 + w
    y1 = y0 + h

    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(frame.shape[1], x1)
    fy1 = min(frame.shape[0], y1)

    if fx0 >= fx1 or fy0 >= fy1:
        return

    ox0 = fx0 - x0
    oy0 = fy0 - y0
    ox1 = ox0 + (fx1 - fx0)
    oy1 = oy0 + (fy1 - fy0)

    patch = rotated[oy0:oy1, ox0:ox1]
    if patch.shape[2] == 4:
        alpha = (patch[:, :, 3].astype(np.float32) / 255.0) * alpha_scale
        rgb = patch[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((patch.shape[0], patch.shape[1]), dtype=np.float32) * alpha_scale
        rgb = patch.astype(np.float32)

    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (dst * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def _apply_netherite_texture_overlay(frame: np.ndarray, body_regions, intensity: float) -> bool:
    global _NETHERITE_TEXTURE_WARNED
    texture = _get_netherite_texture()
    if texture is None:
        if not _NETHERITE_TEXTURE_WARNED:
            print("[visual_effects] No se encontro images/netherite_chestplate.png; usando fallback procedural.")
            _NETHERITE_TEXTURE_WARNED = True
        return False

    if not all(k in body_regions.key_points for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        return False

    h, w = frame.shape[:2]
    ls = _project_point(body_regions.key_points["left_shoulder"], w, h)
    rs = _project_point(body_regions.key_points["right_shoulder"], w, h)
    lh = _project_point(body_regions.key_points["left_hip"], w, h)
    rh = _project_point(body_regions.key_points["right_hip"], w, h)

    shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

    shoulder_w = max(24.0, float(abs(rs[0] - ls[0])))
    torso_h = max(36.0, float(abs(hip_mid[1] - shoulder_mid[1])))

    draw_w = int(shoulder_w * 1.35)
    draw_h = int(torso_h * 1.30)
    cx = int(shoulder_mid[0])
    cy = int(shoulder_mid[1] + draw_h * 0.30)

    angle_deg = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))

    resized = cv2.resize(texture, (max(8, draw_w), max(8, draw_h)), interpolation=cv2.INTER_NEAREST)
    if _MIRROR_CHESTPLATE_TEXTURE:
        resized = cv2.flip(resized, 1)
    if _FLIP_CHESTPLATE_VERTICAL:
        resized = cv2.flip(resized, 0)
    _overlay_rgba_rotated(frame, resized, (cx, cy), angle_deg, alpha_scale=0.95 * intensity)
    return True


def _blend_overlay_with_mask(frame, overlay, mask, alpha):
    alpha_map = ((mask.astype(np.float32) / 255.0) * alpha)[..., None]
    frame[:] = (
        frame.astype(np.float32) * (1.0 - alpha_map) +
        overlay.astype(np.float32) * alpha_map
    ).astype(np.uint8)


def apply_shadow_mode(frame: np.ndarray, body_regions, intensity: float = 1.0):
    if _apply_netherite_texture_overlay(frame, body_regions, intensity):
        return

    region = body_regions.get_netherite_chestplate_region()
    if region is None:
        return

    pts = region["points"]
    inner_pts = region["inner_points"]
    bounds = region["bounds"]

    frame_h, frame_w = frame.shape[:2]
    x1 = min(frame_w - 1, bounds["x"] + bounds["width"] + 4)
    y1 = min(frame_h - 1, bounds["y"] + bounds["height"] + 4)

    plate = np.zeros_like(frame, dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    base_color = (24, 20, 26)
    shadow_color = (14, 12, 16)
    mid_color = (38, 32, 42)
    highlight_color = (58, 50, 62)

    cv2.fillPoly(plate, [pts], base_color, lineType=cv2.LINE_8)
    cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_8)
    cv2.fillPoly(plate, [inner_pts], mid_color, lineType=cv2.LINE_8)

    ls = region["shoulder_left"]
    rs = region["shoulder_right"]
    center_x, shoulder_y = region["center"]
    shoulder_span = max(44, abs(rs[0] - ls[0]))
    attach_h = max(6, int(shoulder_span * 0.10))
    shoulder_left_block = np.array([
        (max(0, ls[0] - int(shoulder_span * 0.24)), max(0, ls[1] - attach_h)),
        (ls[0], max(0, ls[1] - attach_h // 2)),
        (center_x - int(shoulder_span * 0.12), max(0, shoulder_y - int(shoulder_span * 0.08))),
        (center_x - int(shoulder_span * 0.18), shoulder_y + int(shoulder_span * 0.10)),
        (max(0, ls[0] - int(shoulder_span * 0.12)), shoulder_y + int(shoulder_span * 0.06)),
    ], dtype=np.int32)
    shoulder_right_block = np.array([
        (min(frame_w - 1, rs[0] + int(shoulder_span * 0.24)), max(0, rs[1] - attach_h)),
        (rs[0], max(0, rs[1] - attach_h // 2)),
        (center_x + int(shoulder_span * 0.12), max(0, shoulder_y - int(shoulder_span * 0.08))),
        (center_x + int(shoulder_span * 0.18), shoulder_y + int(shoulder_span * 0.10)),
        (min(frame_w - 1, rs[0] + int(shoulder_span * 0.12)), shoulder_y + int(shoulder_span * 0.06)),
    ], dtype=np.int32)
    cv2.fillPoly(plate, [shoulder_left_block], shadow_color, lineType=cv2.LINE_8)
    cv2.fillPoly(plate, [shoulder_right_block], shadow_color, lineType=cv2.LINE_8)

    neck_center_x = int(center_x)
    neck_y = max(0, region["bounds"]["y"] + int(region["bounds"]["height"] * 0.02))
    neck_w = max(10, int(shoulder_span * 0.12))
    collar_pts = np.array([
        (neck_center_x - neck_w, neck_y + 2),
        (neck_center_x - int(neck_w * 0.45), max(0, neck_y - 6)),
        (neck_center_x + int(neck_w * 0.45), max(0, neck_y - 6)),
        (neck_center_x + neck_w, neck_y + 2),
        (neck_center_x + int(neck_w * 0.70), neck_y + 12),
        (neck_center_x - int(neck_w * 0.70), neck_y + 12),
    ], dtype=np.int32)
    cv2.fillPoly(plate, [collar_pts], shadow_color, lineType=cv2.LINE_8)

    left_panel = np.array([
        (center_x - int(shoulder_span * 0.03), neck_y + 16),
        (center_x - int(shoulder_span * 0.22), neck_y + 22),
        (center_x - int(shoulder_span * 0.28), neck_y + int(shoulder_span * 0.28)),
        (center_x - int(shoulder_span * 0.20), neck_y + int(shoulder_span * 0.54)),
        (center_x - int(shoulder_span * 0.06), neck_y + int(shoulder_span * 0.72)),
    ], dtype=np.int32)
    right_panel = np.array([
        (center_x + int(shoulder_span * 0.03), neck_y + 16),
        (center_x + int(shoulder_span * 0.22), neck_y + 22),
        (center_x + int(shoulder_span * 0.28), neck_y + int(shoulder_span * 0.28)),
        (center_x + int(shoulder_span * 0.20), neck_y + int(shoulder_span * 0.54)),
        (center_x + int(shoulder_span * 0.06), neck_y + int(shoulder_span * 0.72)),
    ], dtype=np.int32)
    cv2.fillPoly(plate, [left_panel], (30, 24, 32), lineType=cv2.LINE_8)
    cv2.fillPoly(plate, [right_panel], (30, 24, 32), lineType=cv2.LINE_8)

    cv2.line(plate, (center_x, neck_y + 10), (center_x, y1 - 12), highlight_color, 1, cv2.LINE_8)
    cv2.line(plate, (center_x - int(shoulder_span * 0.12), neck_y + 28), (center_x - int(shoulder_span * 0.12), y1 - 24), (48, 42, 52), 1, cv2.LINE_8)
    cv2.line(plate, (center_x + int(shoulder_span * 0.12), neck_y + 28), (center_x + int(shoulder_span * 0.12), y1 - 24), (48, 42, 52), 1, cv2.LINE_8)

    cv2.polylines(plate, [pts], True, (8, 6, 8), 3, cv2.LINE_8)
    cv2.polylines(plate, [pts], True, (66, 58, 70), 1, cv2.LINE_8)

    bottom_tip = np.array([
        (center_x - int(shoulder_span * 0.12), y1 - int(shoulder_span * 0.10)),
        (center_x + int(shoulder_span * 0.12), y1 - int(shoulder_span * 0.10)),
        (center_x + int(shoulder_span * 0.06), y1),
        (center_x - int(shoulder_span * 0.06), y1),
    ], dtype=np.int32)
    cv2.fillPoly(plate, [bottom_tip], shadow_color, lineType=cv2.LINE_8)

    _blend_overlay_with_mask(frame, plate, mask, alpha=1.0)
