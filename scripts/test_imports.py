"""
test_imports.py
---------------
Script rápido para verificar que todos los módulos nuevos se importan correctamente
antes de ejecutar el programa completo.

Uso (desde la raíz del proyecto):
    python scripts/test_imports.py
"""

import sys
import os

# Añadir la raíz del proyecto al path para poder importar desde src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("Verificando importaciones...")

try:
    print("  [1/8] Importando pose_detector...")
    from src.core.pose_detector import PoseDetector
    print("        ✓ PoseDetector")
    
    print("  [2/8] Importando body_regions...")
    from src.visual.body_regions import BodyRegions
    print("        ✓ BodyRegions")
    
    print("  [3/8] Importando visual_effects...")
    from src.visual.visual_effects import apply_effect_for_gesture, get_effect_name, EFFECT_MAP
    print("        ✓ visual_effects")
    
    print("  [4/8] Importando effect_renderer...")
    from src.visual.effect_renderer import EffectRenderer, draw_effect_status
    print("        ✓ EffectRenderer")
    
    print("  [5/8] Importando gesture_detector...")
    from src.core.gesture_detector import GestureDetector
    print("        ✓ GestureDetector")
    
    print("  [6/8] Importando image_manager...")
    from src.visual.image_manager import ImageManager
    print("        ✓ ImageManager")
    
    print("  [7/8] Importando dataset_collector...")
    from src.data.dataset_collector import DatasetCollector
    print("        ✓ DatasetCollector")
    
    print("  [8/8] Importando dependencias...")
    import cv2
    import mediapipe as mp
    import numpy as np
    print("        ✓ OpenCV, MediaPipe, NumPy")
    
    print("\n" + "="*60)
    print("✅ TODAS LAS IMPORTACIONES EXITOSAS")
    print("="*60)
    
    # Verificar efectos disponibles
    print(f"\n📋 Efectos visuales disponibles ({len(EFFECT_MAP)}):")
    for gesture_id, effect_name in sorted([(k, get_effect_name(k)) for k in EFFECT_MAP.keys()]):
        print(f"   Gesto {gesture_id}: {effect_name}")
    
    print("\n🚀 El sistema está listo para ejecutarse.")
    print("   Ejecuta: python main.py")
    
except ImportError as e:
    print(f"\n❌ ERROR DE IMPORTACIÓN: {e}")
    print("\nVerifica que todas las dependencias estén instaladas:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"\n❌ ERROR INESPERADO: {e}")
