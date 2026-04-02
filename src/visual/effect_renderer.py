"""
effect_renderer.py
------------------
Manages smooth rendering and transitions of visual effects on the user's body.

Features:
  - Smooth fade in/out when effects change
  - Intensity stabilization with temporal smoothing
  - Persistent effect state across frames
  - Configurable transition speeds

This module acts as a layer between gesture detection and visual effects,
ensuring that changes are visually smooth and not jarring.
"""

import time
from typing import Optional

import numpy as np

from .body_regions import BodyRegions
from .visual_effects import apply_effect_for_gesture, get_effect_name


class EffectRenderer:
    """Manages smooth rendering of body effects with transitions."""

    def __init__(
        self,
        fade_speed: float = 0.08,        # fraction per frame (~12 frames to full)
        min_intensity: float = 0.0,
        max_intensity: float = 1.0,
    ):
        self._fade_speed = fade_speed
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity

        # Current state
        self._current_gesture: int = 0
        self._current_intensity: float = 0.0
        self._target_intensity: float = 0.0

        # Transition tracking
        self._transitioning = False
        self._last_gesture_change = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def update_gesture(self, gesture: int):
        """
        Update the target gesture. If different from current, triggers a transition.
        
        Parameters
        ----------
        gesture : int
            The newly detected gesture (0-5, where 0 = no gesture)
        """
        if gesture != self._current_gesture:
            # Gesture changed, start transition
            self._current_gesture = gesture
            self._target_intensity = self._max_intensity if gesture != 0 else 0.0
            self._transitioning = True
            self._last_gesture_change = time.perf_counter()

    def render(
        self,
        frame: np.ndarray,
        body_regions: BodyRegions,
        force_intensity: Optional[float] = None
    ):
        """
        Render the current effect on the frame with smooth transitions.
        
        Parameters
        ----------
        frame : np.ndarray
            BGR image to modify in-place
        body_regions : BodyRegions
            Calculated body regions for effect placement
        force_intensity : float, optional
            If provided, overrides automatic intensity calculation
        """
        # Update intensity (smooth approach to target)
        if self._transitioning:
            self._update_intensity()

        # Determine actual intensity to use
        intensity = force_intensity if force_intensity is not None else self._current_intensity

        # Apply effect if intensity > 0
        if intensity > 0.01 and self._current_gesture != 0:
            apply_effect_for_gesture(
                frame,
                self._current_gesture,
                body_regions,
                intensity
            )

    def get_current_effect_name(self) -> str:
        """Get the name of the currently active effect."""
        if self._current_gesture == 0:
            return "No Effect"
        return get_effect_name(self._current_gesture)

    def get_current_intensity(self) -> float:
        """Get the current effect intensity [0, 1]."""
        return self._current_intensity

    def is_transitioning(self) -> bool:
        """Check if currently transitioning between effects."""
        return self._transitioning

    def reset(self):
        """Reset to no effect."""
        self._current_gesture = 0
        self._current_intensity = 0.0
        self._target_intensity = 0.0
        self._transitioning = False

    # ── private helpers ───────────────────────────────────────────────────────

    def _update_intensity(self):
        """Smoothly approach target intensity."""
        diff = self._target_intensity - self._current_intensity

        if abs(diff) < 0.01:
            # Close enough, snap to target
            self._current_intensity = self._target_intensity
            self._transitioning = False
        else:
            # Move towards target
            if diff > 0:
                # Fade in
                self._current_intensity += self._fade_speed
                self._current_intensity = min(self._current_intensity, self._target_intensity)
            else:
                # Fade out
                self._current_intensity -= self._fade_speed
                self._current_intensity = max(self._current_intensity, self._target_intensity)

        # Clamp to valid range
        self._current_intensity = max(
            self._min_intensity,
            min(self._max_intensity, self._current_intensity)
        )


# ── Debug overlay helper ──────────────────────────────────────────────────────

def draw_effect_status(
    frame: np.ndarray,
    renderer: EffectRenderer,
    position: tuple = (10, 30),
    font_scale: float = 0.6,
    color: tuple = (0, 255, 255),
):
    """
    Draw debug information about the current effect on the frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame to draw on
    renderer : EffectRenderer
        The effect renderer to get status from
    position : tuple
        (x, y) position for text
    font_scale : float
        Font size scale
    color : tuple
        Text color in BGR
    """
    import cv2

    effect_name = renderer.get_current_effect_name()
    intensity = renderer.get_current_intensity()
    transitioning = renderer.is_transitioning()

    text = f"Effect: {effect_name}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)

    # Intensity bar
    bar_x, bar_y = position[0], position[1] + 30
    bar_width, bar_height = 200, 12

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (60, 60, 60), -1)

    # Filled portion
    filled = int(bar_width * intensity)
    bar_color = (0, 200, 255) if not transitioning else (0, 255, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height),
                 bar_color, -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (200, 200, 200), 1)

    # Intensity text
    intensity_text = f"Intensity: {intensity:.2f}"
    cv2.putText(frame, intensity_text, (bar_x + bar_width + 10, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
