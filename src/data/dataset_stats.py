"""
dataset_stats.py
----------------
Muestra un resumen del dataset guardado en dataset_landmarks.csv.

Uso
---
    python dataset_stats.py
    python dataset_stats.py mi_dataset.csv

Imprime
-------
  - Número de muestras por clase
  - Total
  - Clases que faltan o están desbalanceadas
  - Columnas con NaN / valores fuera de rango
"""

import csv
import math
import os
import sys
from collections import Counter

# Navegar a la raíz del proyecto (dos niveles arriba: src/data -> src -> raíz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_CSV  = os.path.join(DATASET_DIR, "dataset_landmarks.csv")

TARGET_PER_CLASS = 2000   # objetivo recomendado por clase


def load_csv(path: str):
    rows, header = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for row in reader:
            if row:
                rows.append(row)
    return header, rows


def check_values(rows: list, n_features: int = 63):
    """Detecta filas con valores anómalos."""
    bad_rows = 0
    for row in rows:
        if len(row) != n_features + 1:
            bad_rows += 1
            continue
        try:
            vals = [float(v) for v in row[1:]]
            if any(math.isnan(v) or abs(v) > 50 for v in vals):
                bad_rows += 1
        except ValueError:
            bad_rows += 1
    return bad_rows


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not os.path.isfile(csv_path):
        print(f"[ERROR] No se encontró el archivo: {csv_path}")
        print("  Ejecuta main.py en modo RECORD para generar datos primero.")
        sys.exit(1)

    header, rows = load_csv(csv_path)
    total = len(rows)

    if total == 0:
        print("[WARNING] El archivo existe pero no tiene filas.")
        sys.exit(0)

    # ── Conteo por clase ──────────────────────────────────────────────────────
    counts = Counter(row[0] for row in rows)
    max_count = max(counts.values())

    print(f"\n{'═'*46}")
    print(f"  Dataset: {os.path.basename(csv_path)}")
    print(f"  Total de muestras: {total}")
    print(f"{'─'*46}")
    print(f"  {'Clase':<10} {'Muestras':>8}  {'Barra':}")
    print(f"{'─'*46}")

    all_classes = ["ONE", "TWO", "THREE", "FOUR", "OK"]
    for cls in all_classes:
        n = counts.get(cls, 0)
        bar_len = int(30 * n / max(max_count, 1))
        bar = "█" * bar_len
        flag = ""
        if n == 0:
            flag = "  ← FALTA"
        elif n < TARGET_PER_CLASS * 0.3:
            flag = "  ← muy pocas"
        elif n < TARGET_PER_CLASS * 0.7:
            flag = "  ← incompleta"
        print(f"  {cls:<10} {n:>8}  {bar}{flag}")

    # clases extra no esperadas
    for cls, n in counts.items():
        if cls not in all_classes:
            print(f"  {cls:<10} {n:>8}  (clase inesperada)")

    print(f"{'─'*46}")
    print(f"  Objetivo por clase: {TARGET_PER_CLASS}")

    # ── Calidad de datos ──────────────────────────────────────────────────────
    bad = check_values(rows)
    print(f"\n  Filas con valores anómalos: {bad}")

    # ── Balance ───────────────────────────────────────────────────────────────
    present = [counts[c] for c in all_classes if counts.get(c, 0) > 0]
    if len(present) > 1:
        ratio = min(present) / max(present)
        balance_ok = ratio >= 0.7
        print(f"  Ratio min/max de clases:    {ratio:.2f}  "
              f"{'✓ bien balanceado' if balance_ok else '✗ desbalanceado'}")

    # ── Consejo ───────────────────────────────────────────────────────────────
    missing = TARGET_PER_CLASS * len(all_classes) - total
    if missing > 0:
        print(f"\n  Faltan ~{missing} muestras para alcanzar el objetivo.")
    else:
        print(f"\n  ¡Dataset completo! Listo para entrenar en Fase 3.")

    print(f"{'═'*46}\n")


if __name__ == "__main__":
    main()
