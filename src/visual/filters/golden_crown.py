"""Golden Crown (Gesture 5) filter with FaceMesh and pose fallback."""

import os

import cv2
import mediapipe as mp
import numpy as np

_CROWN_TEXTURE = None
_CROWN_TEXTURE_LOADED = False
_CROWN_TEXTURE_WARNED = False
_FACE_MESH = None
_FACE_MESH_WARNED = False


def _get_crown_texture() -> np.ndarray | None:
    global _CROWN_TEXTURE, _CROWN_TEXTURE_LOADED
    if _CROWN_TEXTURE_LOADED:
        return _CROWN_TEXTURE

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    candidates = [
        os.path.join(base_dir, "images", "corona_transparent.png"),
        os.path.join(base_dir, "images", "gold-crown-isolated-on-transparent-background.png"),
        os.path.join(base_dir, "images", "crown.png"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            tex = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if tex is not None and tex.size > 0 and tex.shape[2] == 4:
                _CROWN_TEXTURE = tex
                break

    _CROWN_TEXTURE_LOADED = True
    return _CROWN_TEXTURE


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        face_mesh_cls = None
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                face_mesh_cls = mp.solutions.face_mesh.FaceMesh
        except Exception:
            face_mesh_cls = None

        if face_mesh_cls is None:
            try:
                from mediapipe.python.solutions import face_mesh as mp_face_mesh
                face_mesh_cls = mp_face_mesh.FaceMesh
            except Exception:
                face_mesh_cls = None

        if face_mesh_cls is None:
            return None

        _FACE_MESH = face_mesh_cls(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH


def _overlay_png(background: np.ndarray, overlay: np.ndarray, x: int, y: int):
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    roi = background[y1:y2, x1:x2]
    src = overlay[oy1:oy2, ox1:ox2]

    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    rgb = src[:, :, :3].astype(np.float32)
    blended = (rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background


def _apply_crown_face_mesh_filter(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    global _CROWN_TEXTURE_WARNED, _FACE_MESH_WARNED

    crown = _get_crown_texture()
    if crown is None:
        if not _CROWN_TEXTURE_WARNED:
            print("[visual_effects] No se encontro una corona PNG con alpha en images/ para el efecto OK.")
            _CROWN_TEXTURE_WARNED = True
        return frame, False

    face_mesh = _get_face_mesh()
    if face_mesh is None:
        if not _FACE_MESH_WARNED:
            print("[visual_effects] FaceMesh no disponible en esta version de MediaPipe; usando fallback por pose.")
            _FACE_MESH_WARNED = True
        return frame, False

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame, False

    lm = results.multi_face_landmarks[0].landmark
    left = lm[234]
    right = lm[454]
    top = lm[10]

    crown_w = int(abs(right.x - left.x) * w * 1.35)
    if crown_w < 8:
        return frame, False
    crown_h = int(crown_w * crown.shape[0] / crown.shape[1])
    crown_resized = cv2.resize(crown, (crown_w, crown_h), interpolation=cv2.INTER_LINEAR)

    cx = int((left.x + right.x) * 0.5 * w)
    cy = int(top.y * h)
    x = cx - crown_w // 2
    y = cy - int(crown_h * 0.85)

    out = frame.copy()
    _overlay_png(out, crown_resized, x, y)
    return out, True


def _apply_crown_pose_fallback(frame: np.ndarray, body_regions) -> tuple[np.ndarray, bool]:
    crown = _get_crown_texture()
    if crown is None:
        return frame, False

    head = body_regions.get_head_region() if body_regions is not None else None
    if head is None:
        return frame, False

    center = head["center"]
    radius = max(18, int(head["radius"]))
    crown_w = int(radius * 2.6)
    crown_h = int(crown_w * crown.shape[0] / crown.shape[1])
    crown_resized = cv2.resize(crown, (max(8, crown_w), max(8, crown_h)), interpolation=cv2.INTER_LINEAR)

    x = int(center[0] - crown_w / 2)
    y = int(center[1] - crown_h * 1.05)

    out = frame.copy()
    _overlay_png(out, crown_resized, x, y)
    return out, True


def apply_golden_crown(frame, body_regions, intensity: float = 1.0):
    filtered, applied = _apply_crown_face_mesh_filter(frame)
    if not applied:
        filtered, applied = _apply_crown_pose_fallback(frame, body_regions)
    if not applied:
        return

    a = float(np.clip(intensity, 0.0, 1.0))
    frame[:] = cv2.addWeighted(filtered, a, frame, 1.0 - a, 0)
