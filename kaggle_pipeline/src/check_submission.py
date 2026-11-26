"""
Verificador estricto de archivos submission.csv antes de subir a Kaggle (y constructor de submissions).

Chequeos (falla -> exit 1):
  - exactamente 646 800 filas (264 h evaluadas 00,03,..,21 UTC de las 792 h de test × 70 × 35) y columnas EXACTAS en orden: id,RH,T,U,V
  - ids idénticos y en el mismo orden que sample_submission.csv, sin duplicados
  - columnas de predicción numéricas, sin NaN ni ±inf, no constantes
Avisos (no fallan): rangos físicos sospechosos (RH fuera de [0,100], T fuera de [220,320] K, |U|,|V| > 40 m/s).

Uso:
  python kaggle_pipeline/src/check_submission.py --sub submission.csv [--sample data/sample_submission.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import data_file  # noqa: E402

EXPECTED_ROWS = 646800
EXPECTED_COLS = ["id", "RH", "T", "U", "V"]
PHYS_RANGES = {"RH": (0.0, 100.0), "T": (220.0, 320.0), "U": (-40.0, 40.0), "V": (-40.0, 40.0)}


def check_submission_df(sub: pd.DataFrame, sample: pd.DataFrame, expected_rows: int = EXPECTED_ROWS):
    """Devuelve (errores, avisos) como listas de strings."""
    errors, warnings = [], []
    if len(sub) != expected_rows:
        errors.append(f"Filas: {len(sub)} (se esperan {expected_rows}).")
    if len(sample) != expected_rows:
        errors.append(f"sample_submission tiene {len(sample)} filas (se esperan {expected_rows}): ¿archivo correcto?")
    if list(sub.columns) != EXPECTED_COLS:
        errors.append(f"Columnas {list(sub.columns)} != {EXPECTED_COLS} (nombres y orden exactos).")
    if list(sample.columns) != EXPECTED_COLS:
        errors.append(f"sample_submission tiene columnas {list(sample.columns)}.")

    if "id" in sub.columns:
        if sub["id"].duplicated().any():
            errors.append(f"{int(sub['id'].duplicated().sum())} ids duplicados.")
        if len(sub) == len(sample) and "id" in sample.columns:
            same = sub["id"].astype(str).values == sample["id"].astype(str).values
            if not same.all():
                first = int(np.argmin(same))
                errors.append(f"{int((~same).sum())} ids distintos/desordenados vs sample (1.º en fila {first}: "
                              f"{sub['id'].iloc[first]} vs {sample['id'].iloc[first]}).")

    n_null = int(sub.isna().sum().sum())
    if n_null:
        errors.append(f"{n_null} valores NaN/nulos.")
    for col in EXPECTED_COLS[1:]:
        if col not in sub.columns:
            continue
        if not pd.api.types.is_numeric_dtype(sub[col]):
            errors.append(f"Columna '{col}' no es numérica ({sub[col].dtype}).")
            continue
        vals = sub[col].to_numpy(dtype=np.float64)
        if np.isinf(vals).any():
            errors.append(f"Columna '{col}' contiene ±inf.")
        if sub[col].nunique() <= 1:
            errors.append(f"Columna '{col}' es constante ({sub[col].iloc[0]}).")
        lo, hi = PHYS_RANGES[col]
        fin = vals[np.isfinite(vals)]
        if fin.size and (fin.min() < lo - 1e-6 or fin.max() > hi + 1e-6):
            warnings.append(f"'{col}' fuera de rango físico [{lo},{hi}]: min={fin.min():.3f} max={fin.max():.3f}")
    return errors, warnings


def check_submission(sub_path: Path, sample_path: Path, expected_rows: int = EXPECTED_ROWS) -> bool:
    sub_path, sample_path = Path(sub_path), Path(sample_path)
    for p, lbl in ((sub_path, "submission"), (sample_path, "sample_submission")):
        if not p.exists():
            print(f"[ERROR] No existe el archivo {lbl}: {p}")
            return False
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    errors, warnings = check_submission_df(sub, sample, expected_rows)

    print("=" * 60)
    if errors:
        print("VALIDACIÓN DE SUBMISSION FALLIDA — NO SUBIR A KAGGLE")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print(f"SUBMISSION VÁLIDO: {sub_path.name}")
        print(f"   Filas: {len(sub)} | Columnas: {list(sub.columns)}")
        for col in EXPECTED_COLS[1:]:
            print(f"   [{col}] min={sub[col].min():.4f} | mean={sub[col].mean():.4f} | max={sub[col].max():.4f}")
    for w in warnings:
        print(f"  [AVISO] {w}")
    print("=" * 60)
    return not errors


def build_submission(test_preds: np.ndarray, test_times, sample: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """test_preds (N_test, 70, 35, 4) en el orden de test_times -> DataFrame con el orden de sample."""
    pos = {str(t): i for i, t in enumerate(np.asarray(test_times).astype(str))}
    parts = sample["id"].str.split("_", expand=True)
    ti = parts[0].map(pos)
    if ti.isna().any():
        raise ValueError(f"{int(ti.isna().sum())} ids del sample con tiempo ausente en test_times")
    ti = ti.values.astype(int)
    r = parts[1].astype(int).values
    c = parts[2].astype(int).values
    sub = pd.DataFrame({"id": sample["id"]})
    for k, v in enumerate(EXPECTED_COLS[1:]):
        sub[v] = test_preds[ti, r, c, k].astype(np.float64).round(decimals)
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica un submission.csv contra sample_submission.csv")
    parser.add_argument("--sub", type=Path, required=True, help="Ruta al submission.csv generado")
    parser.add_argument("--sample", type=Path, default=None, help="sample_submission.csv (default: data/ o PASSWORD_DATA_DIR)")
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()
    ok = check_submission(args.sub, args.sample or data_file("sample_submission.csv"), args.expected_rows)
    sys.exit(0 if ok else 1)
