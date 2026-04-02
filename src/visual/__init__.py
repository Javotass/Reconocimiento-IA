"""
src.visual
----------
Módulos de efectos visuales y renderizado.

Exports:
  - BodyRegions: Cálculo de regiones corporales
  - EffectRenderer: Renderizado suavizado de efectos
  - ImageManager: Gestión de imágenes de gestos
  - apply_effect_for_gesture: Aplicar efecto según gesto
  - get_effect_name: Obtener nombre del efecto
  - draw_effect_status: Dibujar estado del efecto (debug)
"""

from .body_regions import BodyRegions
from .effect_renderer import EffectRenderer, draw_effect_status
from .image_manager import ImageManager
from .visual_effects import apply_effect_for_gesture, get_effect_name, EFFECT_MAP

__all__ = [
    "BodyRegions",
    "EffectRenderer",
    "ImageManager",
    "apply_effect_for_gesture",
    "get_effect_name",
    "draw_effect_status",
    "EFFECT_MAP",
]
