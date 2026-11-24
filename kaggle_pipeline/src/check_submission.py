"""
Verificador estricto de archivos submission.csv antes de subir a Kaggle.
Evita gastar intentos diarios con archivos corruptos, NaNs, IDs desordenados o columnas erróneas.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def check_submission(sub_path: Path, sample_path: Path) -> bool:
    if not sub_path.exists():
        print(f"[ERROR] No existe el archivo de submission: {sub_path}")
        return False
    if not sample_path.exists():
        print(f"[ERROR] No existe el archivo sample_submission: {sample_path}")
        return False

    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)

    errors = []

    # 1. Dimensiones
    if sub.shape != sample.shape:
        errors.append(f"Shape incorrecto: submission tiene {sub.shape}, pero sample_submission exige {sample.shape}.")

    # 2. Columnas exactas y en el mismo orden
    if list(sub.columns) != list(sample.columns):
        errors.append(f"Columnas no coinciden: {list(sub.columns)} vs {list(sample.columns)}.")

    # 3. Valores nulos o infinitos
    null_count = int(sub.isna().sum().sum())
    if null_count > 0:
        errors.append(f"El submission contiene {null_count} valores NaN/nulos.")

    num_cols = sub.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0 and np.isinf(sub[num_cols].values).any():
        errors.append("El submission contiene valores infinitos (inf / -inf).")

    # 4. Alineación de la columna ID (primera columna)
    if len(sub.columns) > 0 and len(sub) == len(sample):
        id_col = sample.columns[0]
        if id_col in sub.columns and not (sub[id_col].values == sample[id_col].values).all():
            errors.append(f"La columna ID '{id_col}' no está en el mismo orden o no coincide con sample_submission.")

    # 5. Constantes sospechosas (todas las predicciones iguales)
    for col in sample.columns[1:]:
        if col in sub.columns and sub[col].nunique() <= 1:
            errors.append(f"ALERTA: La columna de predicción '{col}' tiene un único valor constante ({sub[col].iloc[0]}).")

    if errors:
        print("=" * 60)
        print("❌ VALIDACIÓN DE SUBMISSION FALLIDA — NO SUBIR A KAGGLE")
        for err in errors:
            print(f"  - {err}")
        print("=" * 60)
        return False

    print("=" * 60)
    print(f"✅ SUBMISSION VÁLIDO: {sub_path.name}")
    print(f"   Filas: {len(sub)} | Columnas: {list(sub.columns)}")
    for col in sub.columns[1:]:
        if pd.api.types.is_numeric_dtype(sub[col]):
            print(f"   [{col}] min={sub[col].min():.4f} | mean={sub[col].mean():.4f} | max={sub[col].max():.4f}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica un submission.csv contra sample_submission.csv")
    parser.add_argument("--sub", type=Path, required=True, help="Ruta al archivo submission.csv generado")
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_submission.csv",
        help="Ruta al sample_submission.csv oficial",
    )
    args = parser.parse_args()
    ok = check_submission(args.sub, args.sample)
    sys.exit(0 if ok else 1)
