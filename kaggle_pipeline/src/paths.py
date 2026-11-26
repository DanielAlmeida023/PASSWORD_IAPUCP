"""
Rutas centralizadas del paquete kaggle_pipeline/src (sin rutas absolutas).

Variables de entorno (opcionales):
  PASSWORD_DATA_DIR : carpeta con train.npz/test.npz/static.npz/sample_submission.csv
                      (Kaggle: /kaggle/input/hackathon-ia-pucp-2026). Si los .npz están en una
                      subcarpeta, se buscan recursivamente.
  PASSWORD_OUT_DIR  : carpeta de salida para OOF, registro.csv, pred_test, state_dicts
                      (Kaggle: /kaggle/working).
Fallback: <repo>/data y <repo>/kaggle_pipeline/experiments.
"""

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent
ROOT_DIR = PIPELINE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"


def _resolve_data_dir() -> Path:
    env = os.environ.get("PASSWORD_DATA_DIR")
    if not env:
        return ROOT_DIR / "data"
    base = Path(env)
    if (base / "train.npz").exists() or not base.exists():
        return base
    hits = sorted(base.rglob("train.npz"))
    return hits[0].parent if hits else base


DATA_DIR = _resolve_data_dir()
OUT_DIR = Path(os.environ.get("PASSWORD_OUT_DIR") or (PIPELINE_DIR / "experiments"))
EXPERIMENTS_DIR = OUT_DIR
OOF_DIR = OUT_DIR / "oof"
REGISTRO_PATH = OUT_DIR / "registro.csv"


def data_file(name: str, data_dir=None) -> Path:
    """Ruta a un archivo de data (train.npz, sample_submission.csv, ...), buscando en subcarpetas."""
    base = Path(data_dir) if data_dir else DATA_DIR
    p = base / name
    if p.exists() or not base.exists():
        return p
    hits = sorted(base.rglob(name))
    return hits[0] if hits else p
