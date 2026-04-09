"""Effect dispatcher and names for gesture-driven visual filters."""

from .filters import (
    apply_armor_overlay,
    apply_fire_shoulders,
    apply_golden_crown,
    apply_ice_shield,
    apply_shadow_mode,
)

EFFECT_MAP = {
    1: apply_shadow_mode,
    2: apply_armor_overlay,
    3: apply_fire_shoulders,
    4: apply_ice_shield,
    5: apply_golden_crown,
}

EFFECT_NAMES = {
    1: "Shadow Mode",
    2: "Golden Hour",
    3: "Fire Shoulders",
    4: "Confetti Rain",
    5: "Golden Crown",
}


def apply_effect_for_gesture(frame, gesture: int, body_regions, intensity: float = 1.0):
    """Apply the visual effect corresponding to the detected gesture."""
    effect_func = EFFECT_MAP.get(gesture)
    if effect_func is not None:
        effect_func(frame, body_regions, intensity)


def get_effect_name(gesture: int) -> str:
    """Get the display name of the effect for a gesture."""
    return EFFECT_NAMES.get(gesture, "No Effect")
