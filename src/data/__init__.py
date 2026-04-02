"""
src.data
--------
Módulos de gestión de datos y datasets.

Exports:
  - DatasetCollector: Captura de landmarks para dataset
  - GESTURE_NAMES: Mapeo de gestos a nombres
"""

from .dataset_collector import DatasetCollector, GESTURE_NAMES

__all__ = ["DatasetCollector", "GESTURE_NAMES"]
