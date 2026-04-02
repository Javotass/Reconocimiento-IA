"""
image_manager.py
----------------
Loads and manages the images associated with each detected gesture.

Expected files (configurable via IMAGE_PATHS):
    images/gesture_1.jpg  →  shown when 1 finger is detected
    images/gesture_2.jpg  →  shown when 2 fingers are detected
    images/gesture_3.jpg  →  shown when 3 fingers are detected

Both .jpg and .png extensions are supported. If an image is missing a
coloured placeholder is generated automatically so the application always
has something to display.
"""

import os
import cv2
import numpy as np

# ── paths (relative to this file's directory) ─────────────────────────────────
# Navegar a la raíz del proyecto (dos niveles arriba: src/visual -> src -> raíz)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMAGES_DIR = os.path.join(_PROJECT_ROOT, "images")

IMAGE_PATHS: dict[int, str] = {
    1: os.path.join(_IMAGES_DIR, "gesture_1.jpg"),
    2: os.path.join(_IMAGES_DIR, "gesture_2.jpg"),
    3: os.path.join(_IMAGES_DIR, "gesture_3.jpg"),
    4: os.path.join(_IMAGES_DIR, "gesture_4.jpg"),
    5: os.path.join(_IMAGES_DIR, "gesture_5.jpg"),
}

# Placeholder colours (BGR) used when a file is missing
_PLACEHOLDER_COLORS: dict[int, tuple] = {
    1: (50,  180,  80),   # green
    2: (200,  80,  50),   # blue-ish
    3: (50,   80, 200),   # red-ish
    4: (20,  120, 200),   # amber (4 fingers)
    5: (30,  160, 210),   # teal (OK)
}

# Target display size for all gesture images
DISPLAY_SIZE = (480, 480)   # (width, height)


class ImageManager:
    """Loads gesture images at startup and serves resized copies on demand."""

    def __init__(self, display_size: tuple[int, int] = DISPLAY_SIZE):
        self._size = display_size           # (w, h)
        self._cache: dict[int, np.ndarray] = {}
        self._load_all()

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, gesture: int) -> np.ndarray:
        """
        Return the BGR image for *gesture* (1, 2 or 3), already resized.
        If gesture is 0 / unknown, return a neutral dark frame.
        """
        return self._cache.get(gesture, self._blank())

    def reload(self):
        """Force-reload all images from disk (useful during development)."""
        self._cache.clear()
        self._load_all()

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_all(self):
        for gesture, path in IMAGE_PATHS.items():
            # Try exact path first, then common alternative extensions
            found_path = self._find_file(path)
            if found_path:
                img = cv2.imread(found_path)
                if img is not None:
                    self._cache[gesture] = self._resize(img)
                    print(f"[ImageManager] Loaded gesture {gesture}: {found_path}")
                    continue
            # Fallback: generate a placeholder
            print(
                f"[ImageManager] WARNING – image not found for gesture {gesture} "
                f"({path}). Using placeholder."
            )
            self._cache[gesture] = self._placeholder(gesture)

    @staticmethod
    def _find_file(path: str) -> str | None:
        """Return *path* if it exists, else try .png / .jpg swap, else None."""
        if os.path.isfile(path):
            return path
        alt = path.replace(".jpg", ".png") if path.endswith(".jpg") else path.replace(".png", ".jpg")
        return alt if os.path.isfile(alt) else None

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, self._size, interpolation=cv2.INTER_AREA)

    def _placeholder(self, gesture: int) -> np.ndarray:
        w, h = self._size
        color = _PLACEHOLDER_COLORS.get(gesture, (80, 80, 80))
        canvas = np.full((h, w, 3), color, dtype=np.uint8)
        # Draw finger count as big text
        text  = str(gesture)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 8
        thick = 12
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = (w - tw) // 2
        ty = (h + th) // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        # Small subtitle
        sub = f"Gesto {gesture} – reemplaza images/gesture_{gesture}.jpg"
        cv2.putText(canvas, sub, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def _blank(self) -> np.ndarray:
        w, h = self._size
        canvas = np.full((h, w, 3), 30, dtype=np.uint8)
        cv2.putText(canvas, "Sin gesto", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas
