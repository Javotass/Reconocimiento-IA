"""Filter modules grouped by gesture effect."""

from .shadow_mode import apply_shadow_mode
from .golden_hour import apply_armor_overlay
from .fire_shoulders import apply_fire_shoulders
from .confetti_rain import apply_ice_shield
from .golden_crown import apply_golden_crown

__all__ = [
    "apply_shadow_mode",
    "apply_armor_overlay",
    "apply_fire_shoulders",
    "apply_ice_shield",
    "apply_golden_crown",
]
