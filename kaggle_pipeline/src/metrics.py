"""
Métrica compartida idéntica a Kaggle (Hackathon IA PUCP 2026 · Equipo PASSWORD).

    score = 1 - mean_v( RMSE_v / RMSE_clim_v ),   v ∈ {RH, T, U, V}

- La climatología de referencia es la media por hora UTC (24 mapas 70x35x4) calculada
  SOLO con el train disponible (en CV: el train del fold; en Kaggle: todo train.npz).
- Kaggle evalúa SOLO las horas UTC 00,03,...,21 -> por defecto se filtra a esas horas.
- RMSE_v se calcula sobre todas las muestras y píxeles evaluados (no promedio de RMSE por hora).

Usar SIEMPRE `kaggle_like_score()` para reportar CV; `evaluate_skill_score()` de train.py
(todas las horas, clim_rmse precomputado) queda solo para selección interna de épocas.
"""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

VARS = ["RH", "T", "U", "V"]
KAGGLE_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)


def hours_utc(times: Sequence) -> np.ndarray:
    """Hora UTC (int) desde strings/ints 'YYYYMMDDHH'."""
    return np.array([int(str(t)[-2:]) for t in np.asarray(times).ravel()], dtype=np.int64)


def kaggle_hour_mask(times: Sequence, hours: Iterable[int] = KAGGLE_HOURS) -> np.ndarray:
    return np.isin(hours_utc(times), np.asarray(tuple(hours), dtype=np.int64))


def clim_by_hour(Y: np.ndarray, times: Sequence) -> np.ndarray:
    """
    Climatología por hora UTC: (24, 70, 35, 4) float32. Las horas sin muestras quedan en NaN
    (kaggle_like_score lanza error si se intenta evaluar una hora sin climatología).
    Acumula en float64 para no perder precisión con T ~ 290 K.
    """
    h = hours_utc(times)
    if len(h) != len(Y):
        raise ValueError(f"len(times)={len(h)} != len(Y)={len(Y)}")
    out = np.full((24,) + Y.shape[1:], np.nan, dtype=np.float64)
    for k in range(24):
        m = h == k
        if m.any():
            out[k] = Y[m].astype(np.float64).mean(axis=0)
    return out.astype(np.float32)


def _rmse_per_var(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.astype(np.float64) - b.astype(np.float64)
    return np.sqrt((d * d).reshape(-1, d.shape[-1]).mean(axis=0))


def score_with_clim(
    pred: np.ndarray,
    y: np.ndarray,
    times: Sequence,
    clim: np.ndarray,
    hours: Optional[Iterable[int]] = KAGGLE_HOURS,
) -> Dict:
    """Igual que kaggle_like_score pero con climatología (24, H, W, 4) ya calculada."""
    if pred.shape != y.shape:
        raise ValueError(f"pred {pred.shape} != y {y.shape}")
    h = hours_utc(times)
    if len(h) != len(y):
        raise ValueError(f"len(times)={len(h)} != len(y)={len(y)}")
    mask = np.ones(len(h), dtype=bool) if hours is None else np.isin(h, np.asarray(tuple(hours)))
    if not mask.any():
        raise ValueError("Ninguna muestra cae en las horas evaluadas.")
    h_eval = h[mask]
    clim_eval = clim[h_eval]
    if np.isnan(clim_eval).any():
        missing = sorted(set(int(k) for k in h_eval if np.isnan(clim[k]).any()))
        raise ValueError(f"Climatología sin datos para horas UTC {missing}")

    rmse = _rmse_per_var(pred[mask], y[mask])
    rmse_clim = _rmse_per_var(clim_eval, y[mask])
    skill = 1.0 - rmse / rmse_clim
    return {
        "score": float(skill.mean()),
        "skill": {v: float(s) for v, s in zip(VARS, skill)},
        "rmse": {v: float(r) for v, r in zip(VARS, rmse)},
        "rmse_clim": {v: float(r) for v, r in zip(VARS, rmse_clim)},
        "n_eval": int(mask.sum()),
    }


def kaggle_like_score(
    pred: np.ndarray,
    y: np.ndarray,
    times: Sequence,
    clim_ref_Y: np.ndarray,
    clim_ref_times: Sequence,
    hours: Optional[Iterable[int]] = KAGGLE_HOURS,
) -> Dict:
    """
    pred, y        : (N, 70, 35, 4) predicción y verdad de las horas a evaluar (p. ej. val del fold).
    times          : (N,) 'YYYYMMDDHH' de pred/y.
    clim_ref_Y     : Y del train del fold (para la climatología de referencia por hora UTC).
    clim_ref_times : times de clim_ref_Y.
    hours          : horas UTC evaluadas (default Kaggle 00,03,...,21; None = todas).

    Devuelve dict: score, skill{var}, rmse{var}, rmse_clim{var}, n_eval.
    """
    clim = clim_by_hour(clim_ref_Y, clim_ref_times)
    return score_with_clim(pred, y, times, clim, hours=hours)


def format_score(res: Dict) -> str:
    sk = " ".join(f"{v}={res['skill'][v]:.4f}" for v in VARS)
    return f"score={res['score']:.4f} | {sk} | n={res['n_eval']}"
