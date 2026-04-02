"""
train_model.py
--------------
Fase 3 — Entrenamiento del clasificador de gestos.

Pasos que ejecuta
─────────────────
  1. Carga el CSV de landmarks
  2. Limpia filas anómalas y elimina duplicados exactos
  3. Balancea las clases (undersample a la clase mínima)
  4. Divide en train / val / test  (70 / 15 / 15 %)
  5. Entrena Random Forest + MLP
  6. Evalúa ambos modelos (accuracy, matriz de confusión, precision/recall)
  7. Guarda el mejor modelo en  dataset/gesture_model.pkl
               y el codificador en  dataset/label_encoder.pkl

Uso
───
    python train_model.py
    python train_model.py --csv dataset/mi_dataset.csv
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# ── rutas ─────────────────────────────────────────────────────────────────────
# Navegar a la raíz del proyecto (un nivel arriba: scripts -> raíz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_CSV  = os.path.join(DATASET_DIR, "dataset_landmarks.csv")
MODEL_PATH   = os.path.join(DATASET_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(DATASET_DIR, "label_encoder.pkl")

# ── hiperparámetros ───────────────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                 n_jobs=-1, random_state=42)
MLP_PARAMS = dict(hidden_layer_sizes=(128, 64), activation="relu",
                  max_iter=500, random_state=42, early_stopping=True,
                  validation_fraction=0.1)

LABEL_ORDER = ["ONE", "TWO", "THREE", "FOUR", "OK"]   # orden deseado
N_FEATURES  = 63   # 21 landmarks × 3 coords


# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Carga el CSV, elimina filas anómalas y duplicados exactos."""
    df = pd.read_csv(csv_path)
    original = len(df)

    # Verificar estructura
    if df.shape[1] != N_FEATURES + 1:
        sys.exit(f"[ERROR] Se esperaban {N_FEATURES + 1} columnas, "
                 f"hay {df.shape[1]}.")

    # Renombrar primera columna a 'label' si hace falta
    df.columns = ["label"] + list(df.columns[1:])

    # Eliminar filas con NaN
    df.dropna(inplace=True)

    # Eliminar filas con valores fuera de rango (pueden ser errores de grabación)
    feature_cols = df.columns[1:]
    mask_ok = (df[feature_cols].abs() <= 50).all(axis=1)
    df = df[mask_ok]

    # Eliminar duplicados exactos
    df.drop_duplicates(inplace=True)

    removed = original - len(df)
    print(f"[Limpieza] {original} → {len(df)} filas  ({removed} eliminadas)")
    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample: todas las clases al tamaño de la menor."""
    counts = df["label"].value_counts()
    min_count = counts.min()
    print(f"\n[Balance] Muestras por clase antes: {dict(counts)}")
    df_bal = (df.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(min_count, random_state=42)))
    print(f"[Balance] Muestras por clase después: "
          f"{dict(df_bal['label'].value_counts())}")
    return df_bal.reset_index(drop=True)


def split_data(df: pd.DataFrame, label_enc: LabelEncoder):
    """Divide en train/val/test (70/15/15) de forma estratificada."""
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = label_enc.transform(df["label"].values)

    # Primera división: train 70 % / temp 30 %
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    # Segunda división: val 15 % / test 15 % (del total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"\n[Split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(name: str, model, X_val, y_val, X_test, y_test,
             label_enc: LabelEncoder):
    """Imprime métricas de validación y test."""
    classes = label_enc.classes_

    val_acc  = accuracy_score(y_val,  model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Accuracy val : {val_acc*100:.1f}%")
    print(f"  Accuracy test: {test_acc*100:.1f}%")

    print("\n  Clasificación por clase (test):")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=classes, digits=3))

    print("  Matriz de confusión (test):")
    cm = confusion_matrix(y_test, model.predict(X_test))
    header = f"{'':8s}" + "".join(f"{c:>8s}" for c in classes)
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {classes[i]:8s}" + "".join(f"{v:>8d}" for v in row))

    return test_acc


def train(csv_path: str):
    print(f"\n{'═'*50}")
    print("  FASE 3 — Entrenamiento del modelo")
    print(f"  Dataset: {csv_path}")
    print(f"{'═'*50}")

    # ── 1. Cargar y limpiar ───────────────────────────────────────────────────
    df = load_and_clean(csv_path)

    # ── 2. Balancear ─────────────────────────────────────────────────────────
    df = balance_classes(df)

    # ── 3. Codificar etiquetas ────────────────────────────────────────────────
    present = [l for l in LABEL_ORDER if l in df["label"].unique()]
    label_enc = LabelEncoder()
    label_enc.fit(present)
    joblib.dump(label_enc, ENCODER_PATH)
    print(f"\n[Encoder] Clases: {list(label_enc.classes_)}")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, label_enc)

    # ── 5. Entrenar ambos modelos ─────────────────────────────────────────────
    print("\n[Entrenando] Random Forest …")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    print("[Entrenando] MLP …")
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)

    # ── 6. Evaluar ────────────────────────────────────────────────────────────
    acc_rf  = evaluate("Random Forest", rf,  X_val, y_val, X_test, y_test, label_enc)
    acc_mlp = evaluate("MLP",           mlp, X_val, y_val, X_test, y_test, label_enc)

    # ── 7. Guardar el mejor ───────────────────────────────────────────────────
    best_model = rf if acc_rf >= acc_mlp else mlp
    best_name  = "Random Forest" if acc_rf >= acc_mlp else "MLP"
    joblib.dump(best_model, MODEL_PATH)

    print(f"\n{'═'*50}")
    print(f"  Mejor modelo: {best_name}  ({max(acc_rf, acc_mlp)*100:.1f}% test)")
    print(f"  Guardado en:  {MODEL_PATH}")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Ruta al CSV de landmarks")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"[ERROR] No se encontró: {args.csv}\n"
                 "  Graba datos primero con python main.py (modo RECORD).")

    train(args.csv)
