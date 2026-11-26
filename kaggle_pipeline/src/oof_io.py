"""
Guardado/carga de predicciones OOF por miembro y fold.

Archivo: experiments/oof/oof_{miembro}_{seed}_fold{k}.npy  (+ _times.npy)
Formato: float16 de (pred − OFFSET), OFFSET = [0, 273.15, 0, 0].
  ¿Por qué el offset? En float16 el paso entre valores cerca de 290 K es 0.25 K
  (error de cuantización RMS ≈ 0.07 K vs RMSE del modelo ≈ 0.83 K -> −0.002 de skill en T).
  Guardando T en °C (|T| < 40) el paso es <= 0.03 K: despreciable. RH (paso <= 0.06 %) y U/V ok.
Usar SIEMPRE load_oof() para leer (re-suma el offset y devuelve float32).
"""

from pathlib import Path
from typing import Tuple

import numpy as np

OOF_OFFSET = np.array([0.0, 273.15, 0.0, 0.0], dtype=np.float32)
from paths import EXPERIMENTS_DIR, OOF_DIR  # env PASSWORD_OUT_DIR


def oof_paths(member: str, seed: int, fold: int, oof_dir: Path = OOF_DIR) -> Tuple[Path, Path]:
    base = Path(oof_dir) / f"oof_{member}_{seed}_fold{fold}"
    return base.with_suffix(".npy"), Path(str(base) + "_times.npy")


def save_oof(pred: np.ndarray, times: np.ndarray, member: str, seed: int, fold: int, oof_dir: Path = OOF_DIR) -> Path:
    p_pred, p_times = oof_paths(member, seed, fold, oof_dir)
    p_pred.parent.mkdir(parents=True, exist_ok=True)
    np.save(p_pred, (pred.astype(np.float32) - OOF_OFFSET).astype(np.float16))
    np.save(p_times, np.asarray(times).astype("U10"))
    return p_pred


def load_oof(member: str, seed: int, fold: int, oof_dir: Path = OOF_DIR) -> Tuple[np.ndarray, np.ndarray]:
    p_pred, p_times = oof_paths(member, seed, fold, oof_dir)
    pred = np.load(p_pred).astype(np.float32) + OOF_OFFSET
    times = np.load(p_times).astype(str)
    return pred, times


def save_pred_test(pred: np.ndarray, times: np.ndarray, member: str, seed: int, out_dir: Path) -> Path:
    """Predicción de test (refit) en float16 con el mismo offset que el OOF."""
    p = Path(out_dir) / f"pred_test_{member}_{seed}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, (pred.astype(np.float32) - OOF_OFFSET).astype(np.float16))
    np.save(Path(out_dir) / f"pred_test_{member}_{seed}_times.npy", np.asarray(times).astype("U10"))
    return p


def load_pred_test(member: str, seed: int, out_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.load(Path(out_dir) / f"pred_test_{member}_{seed}.npy").astype(np.float32) + OOF_OFFSET
    times = np.load(Path(out_dir) / f"pred_test_{member}_{seed}_times.npy").astype(str)
    return pred, times


def load_oof_by_name(name: str, fold: int, oof_dir: Path = OOF_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """name = '{miembro}_{seed}' (p. ej. 'unet_203' o 'fno-m86_304')."""
    member, seed = name.rsplit("_", 1)
    return load_oof(member, int(seed), fold, oof_dir)
