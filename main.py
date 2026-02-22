"""
main.py
-------
Entry point for the real-time hand-gesture → image switcher.

Layout
------
┌────────────────────┬─────────────────────┐
│   Cámara en vivo   │   Imagen del gesto  │
│  (con landmarks)   │   (1 / 2 / 3 / –)  │
└────────────────────┴─────────────────────┘

Controles
---------
  Q  – salir
  R  – recargar imágenes desde disco (útil mientras reemplazas archivos)
  S  – guardar captura de pantalla en `capturas/`
"""

import os
import sys
import time
import cv2
import numpy as np

from gesture_detector import GestureDetector
from image_manager    import ImageManager

# ── configuración ──────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0          # índice de la cámara (0 = predeterminada)
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
PANEL_SIZE      = 480        # tamaño (px) del panel derecho
WINDOW_TITLE    = "Reconocimiento de gestos – Q para salir"

# Colores (BGR)
COLOR_ACTIVE    = (0, 255, 100)   # verde vibrante – gesto activo
COLOR_IDLE      = (180, 180, 180) # gris – sin gesto
COLOR_BG_BANNER = (30, 30, 30)
COLOR_DEBUG     = (80, 200, 255)  # amarillo-cyan – texto debug

# Fade
FADE_STEP = 0.08   # fracción por frame (~12 frames = ~0.4 s a 30 fps)
# ────────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_hud(
    gesture: int, raw_gesture: int, fps: float, frame_w: int,
    debug_mode: bool = False, confidence: float = 1.0, hand_side: str = "?"
) -> np.ndarray:
    """Build a status banner below the camera feed."""
    h   = 38 if not debug_mode else 62
    hud = np.full((h, frame_w, 3), COLOR_BG_BANNER, dtype=np.uint8)

    _LABELS = {1: "Gesto 1", 2: "Gesto 2", 3: "Gesto 3", 4: "Gesto 4", 5: "Gesto OK"}
    if gesture and gesture != 0:
        label = _LABELS.get(gesture, f"Gesto {gesture}")
        color = COLOR_ACTIVE
    else:
        label = "Sin gesto claro"
        color = COLOR_IDLE

    # ── fila principal ────────────────────────────────────────────────────────
    cv2.putText(hud, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(hud, f"FPS: {fps:4.1f}", (frame_w - 130, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 200, 255), 1, cv2.LINE_AA)

    # ── fila debug (solo si debug_mode) ──────────────────────────────────────
    if debug_mode:
        conf_color = COLOR_ACTIVE if confidence >= 0.6 else (0, 140, 255) if confidence >= 0.35 else (0, 60, 220)
        dbg = (f"raw={raw_gesture}  suav={gesture}  mano={hand_side}"
               f"  conf={confidence:.2f}")
        cv2.putText(hud, dbg, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    COLOR_DEBUG, 1, cv2.LINE_AA)
        # barra de confianza
        bar_x, bar_y, bar_w, bar_h = frame_w - 160, 40, 140, 10
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60,60,60), -1)
        filled = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), conf_color, -1)
        cv2.putText(hud, "conf", (bar_x - 42, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DEBUG, 1, cv2.LINE_AA)
    return hud


def build_gesture_panel(img: np.ndarray, gesture: int, panel_size: int) -> np.ndarray:
    """Add a small badge with the gesture number on top of the gesture image."""
    panel = cv2.resize(img, (panel_size, panel_size), interpolation=cv2.INTER_AREA)

    if gesture in (1, 2, 3):
        badge_text = str(gesture)
        font    = cv2.FONT_HERSHEY_SIMPLEX
        scale   = 2.0
        thick   = 4
        (tw, th), _ = cv2.getTextSize(badge_text, font, scale, thick)
        pad = 10
        bx, by = 20, 20
        # semi-transparent rectangle
        overlay = panel.copy()
        cv2.rectangle(overlay, (bx - pad, by - pad),
                      (bx + tw + pad, by + th + pad), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)
        cv2.putText(panel, badge_text, (bx, by + th),
                    font, scale, COLOR_ACTIVE, thick, cv2.LINE_AA)
    return panel


def run():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
        sys.exit(1)

    detector = GestureDetector()
    images   = ImageManager(display_size=(PANEL_SIZE, PANEL_SIZE))
    captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    prev_time  = time.perf_counter()
    fps        = 0.0
    debug_mode = False

    # ── estado fade ──────────────────────────────────────────────────────────
    last_gesture      = -1
    fade_alpha        = 1.0
    fading_from_panel = None
    displayed_panel   = None
    # ─────────────────────────────────────────────────────────────────────────

    print("[INFO] Sistema listo. Muestra 1, 2 o 3 dedos frente a la cámara.")
    print("[INFO]  Q → salir  |  R → recargar imágenes  |  S → guardar captura  |  D → debug")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame vacío, reintentando…")
            continue

        frame = cv2.flip(frame, 1)   # efecto espejo más natural

        # ── detección ────────────────────────────────────────────────────────
        gesture, annotated, raw, confidence, hand_side = detector.process(frame)

        # ── FPS ──────────────────────────────────────────────────────────────
        now  = time.perf_counter()
        fps  = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-9))
        prev_time = now

        # ── panel derecho con fade ───────────────────────────────────────────
        target_img = images.get(gesture)
        target_img = cv2.resize(target_img, (PANEL_SIZE, PANEL_SIZE))

        if gesture != last_gesture:
            # inicio de transición: guardamos el panel que se estaba mostrando
            fading_from_panel = displayed_panel.copy() if displayed_panel is not None else target_img.copy()
            last_gesture       = gesture
            fade_alpha         = 0.0

        fade_alpha = min(1.0, fade_alpha + FADE_STEP)

        if fade_alpha < 1.0 and fading_from_panel is not None:
            displayed_panel = cv2.addWeighted(
                fading_from_panel, 1.0 - fade_alpha,
                target_img,        fade_alpha, 0
            )
        else:
            displayed_panel = target_img

        gesture_panel = build_gesture_panel(displayed_panel, gesture, PANEL_SIZE)

        # ── ajustar alturas ──────────────────────────────────────────────────
        cam_h, cam_w  = annotated.shape[:2]
        panel_h       = PANEL_SIZE

        if cam_h != panel_h:
            scale     = panel_h / cam_h
            new_w     = int(cam_w * scale)
            annotated = cv2.resize(annotated, (new_w, panel_h))
            cam_h, cam_w = annotated.shape[:2]

        # ── HUD ──────────────────────────────────────────────────────────────
        hud          = build_hud(gesture, raw, fps, cam_w + 4 + PANEL_SIZE,
                                  debug_mode, confidence, hand_side)
        separator    = np.full((panel_h, 4, 3), COLOR_BG_BANNER, dtype=np.uint8)
        top_row      = np.hstack([annotated, separator, gesture_panel])
        composite    = np.vstack([top_row, hud])

        cv2.imshow(WINDOW_TITLE, composite)

        # ── teclas y cierre con botón X ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        # botón X de la ventana
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Ventana cerrada.")
            break
        if key == ord('q') or key == 27:
            print("[INFO] Saliendo…")
            break
        elif key == ord('d') or key == ord('D'):
            debug_mode = not debug_mode
            print(f"[INFO] Debug {'activado' if debug_mode else 'desactivado'}.")
        elif key == ord('r'):
            images.reload()
            detector.reset()
            print("[INFO] Imágenes recargadas.")
        elif key == ord('s'):
            ensure_dir(captures_dir)
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(captures_dir, f"captura_{ts}.png")
            cv2.imwrite(path, composite)
            print(f"[INFO] Captura guardada en {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
