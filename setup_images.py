"""
setup_images.py
---------------
Generates visually rich placeholder images for gestures 1, 2 and 3.
Run this once after cloning the project to have ready-to-use images.

Later, simply replace the files in /images/ with your own photos or artwork
and press R inside the running application to reload them without restarting.

Usage
-----
    python setup_images.py
"""

import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

W, H = 480, 480

# ─── per-gesture configuration ───────────────────────────────────────────────
GESTURES = {
    1: {
        "bg_color"   : (30, 120, 50),          # dark green
        "accent"     : (80, 255, 130),
        "label"      : "UNO",
        "emoji_text" : "1",
        "hint"       : "Levanta solo el indice",
    },
    2: {
        "bg_color"   : (120, 50, 30),           # dark blue
        "accent"     : (255, 160, 80),
        "label"      : "DOS",
        "emoji_text" : "2",
        "hint"       : "Levanta indice y corazon",
    },
    3: {
        "bg_color"   : (40, 30, 120),           # dark red-purple
        "accent"     : (100, 130, 255),
        "label"      : "TRES",
        "emoji_text" : "3",
        "hint"       : "Levanta  indice, corazon y anular",
    },
}

font_big   = cv2.FONT_HERSHEY_DUPLEX
font_small = cv2.FONT_HERSHEY_SIMPLEX


def draw_gradient(canvas, color_top, color_bot):
    for y in range(H):
        t = y / H
        color = tuple(int(color_top[c] * (1 - t) + color_bot[c] * t) for c in range(3))
        canvas[y, :] = color


def draw_hand_icon(canvas, n_fingers: int, accent):
    """Draw a simple stylised hand showing n_fingers raised."""
    cx, cy = W // 2, H // 2 + 40  # palm centre

    # palm
    cv2.ellipse(canvas, (cx, cy + 30), (55, 65), 0, 0, 360, accent, -1)

    # finger positions (x_offset, height, draw?)
    FINGERS = [
        (-45, 110, True),    # index
        (-15, 130, True),    # middle
        (15,  125, n_fingers >= 2),
        (45,  105, n_fingers >= 3),
    ]
    for i, (dx, fh, up) in enumerate(FINGERS):
        fx = cx + dx
        if up and i < n_fingers:
            # extended finger
            cv2.rectangle(canvas, (fx - 12, cy - fh), (fx + 12, cy + 10), accent, -1)
            cv2.ellipse(canvas, (fx, cy - fh), (12, 14), 0, 0, 360, accent, -1)
        else:
            # folded finger (small nub)
            cv2.rectangle(canvas, (fx - 12, cy - 30), (fx + 12, cy + 10), accent, -1)


def make_placeholder(gesture: int, cfg: dict) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    bg_dark = tuple(max(0, c - 40) for c in cfg["bg_color"])
    draw_gradient(canvas, cfg["bg_color"], bg_dark)

    # decorative circle
    cv2.circle(canvas, (W // 2, H // 2), 190, cfg["accent"], 3)
    cv2.circle(canvas, (W // 2, H // 2), 185, (255, 255, 255), 1)

    # hand icon in circle
    draw_hand_icon(canvas, gesture, cfg["accent"])

    # big number top-left badge
    cv2.rectangle(canvas, (0, 0), (90, 90), (0, 0, 0), -1)
    cv2.rectangle(canvas, (0, 0), (90, 90), cfg["accent"], 3)
    cv2.putText(canvas, cfg["emoji_text"], (15, 78),
                font_big, 2.8, cfg["accent"], 5, cv2.LINE_AA)

    # label
    (lw, lh), _ = cv2.getTextSize(cfg["label"], font_big, 2.2, 5)
    cv2.putText(canvas, cfg["label"], ((W - lw) // 2, H - 60),
                font_big, 2.2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(canvas, cfg["label"], ((W - lw) // 2, H - 60),
                font_big, 2.2, cfg["accent"], 3, cv2.LINE_AA)

    # hint
    (hw, _), _ = cv2.getTextSize(cfg["hint"], font_small, 0.55, 1)
    cv2.putText(canvas, cfg["hint"], ((W - hw) // 2, H - 28),
                font_small, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # bottom bar
    cv2.rectangle(canvas, (0, H - 10), (W, H), cfg["accent"], -1)

    return canvas


if __name__ == "__main__":
    for gesture, cfg in GESTURES.items():
        img  = make_placeholder(gesture, cfg)
        path = os.path.join(IMAGES_DIR, f"gesture_{gesture}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[setup] Imagen generada: {path}")

    print("\n¡Listo! Puedes reemplazar las imágenes en /images/ con las tuyas propias.")
    print("Formatos soportados: .jpg / .png")
