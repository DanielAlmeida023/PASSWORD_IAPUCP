"""
Post-proceso escalar por hora UTC x variable (Fase 2.7 del plan): y' = a[h, v] * p + b[h, v].

Solo 24 x 4 x 2 escalares (no por píxel). Modos de ajuste (`fit_hourly_affine`):
  - "bias"       : a = 1, b[h, v] = mean(y - p) en la hora h
  - "affine"     : regresión OLS de y sobre p por (h, v), con shrink de a hacia 1 (shrink_a in [0, 1])
  - "affine_var" : a[v], b[v] comunes a todas las horas
Variables fuera de `vars_idx` y horas sin datos quedan en identidad (a=1, b=0).
RH se recorta a [0, 100] tras aplicar (como predict_v7).

Los parámetros se guardan en norm_stats.npz como `pp_a`, `pp_b` (24, 4) float32 (+ `pp_mode` str) y el
mismo `apply_hourly_affine` se usa en el export y en el producto (app/inti_inference.py). Ausentes => identidad.
Uso honesto: estimar SIEMPRE con OOF de folds distintos al evaluado (ver wind_study.py).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

PP_KEYS = ("pp_a", "pp_b")


def _hours(times: Sequence) -> np.ndarray:
    return np.array([int(str(t)[-2:]) for t in np.asarray(times).ravel()], dtype=np.int64)


def identity_params() -> Dict[str, np.ndarray]:
    return {"pp_a": np.ones((24, 4), np.float32), "pp_b": np.zeros((24, 4), np.float32)}


def fit_hourly_affine(pred: np.ndarray, y: np.ndarray, times: Sequence, mode: str = "bias",
                      shrink_a: float = 0.0, vars_idx: Sequence[int] = (0, 1, 2, 3),
                      hours: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """pred, y: (N, H, W, 4) físicos; times (N,). Devuelve {'pp_a', 'pp_b'} (24, 4) float32."""
    if mode not in ("bias", "affine", "affine_var"):
        raise ValueError(f"mode {mode!r}")
    h = _hours(times)
    if hours is not None:
        keep = np.isin(h, np.asarray(tuple(hours)))
        pred, y, h = pred[keep], y[keep], h[keep]
    out = identity_params()
    a, b = out["pp_a"], out["pp_b"]
    groups = [np.arange(len(h))] if mode == "affine_var" else [np.where(h == k)[0] for k in range(24)]
    for gi, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        for v in vars_idx:
            p = pred[idx, ..., v].astype(np.float64).ravel()
            t = y[idx, ..., v].astype(np.float64).ravel()
            if mode == "bias":
                av, bv = 1.0, float((t - p).mean())
            else:
                pm, tm = p.mean(), t.mean()
                var = ((p - pm) ** 2).mean()
                a_ols = float(((p - pm) * (t - tm)).mean() / var) if var > 1e-12 else 1.0
                av = 1.0 + (1.0 - float(shrink_a)) * (a_ols - 1.0)
                bv = float(tm - av * pm)
            if mode == "affine_var":
                a[:, v], b[:, v] = av, bv
            else:
                a[gi, v], b[gi, v] = av, bv
    return out


def apply_hourly_affine(pred: np.ndarray, times: Sequence, params: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Aplica y' = a[h] * p + b[h] por variable; params None o sin claves pp_* => devuelve pred tal cual."""
    if params is None or any(k not in params for k in PP_KEYS):
        return pred
    a = np.asarray(params["pp_a"], np.float32)
    b = np.asarray(params["pp_b"], np.float32)
    h = _hours(times)
    shape = (len(h),) + (1,) * (pred.ndim - 2) + (pred.shape[-1],)
    out = pred * a[h].reshape(shape) + b[h].reshape(shape)
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    return out.astype(np.float32)


def params_from_stats(stats) -> Optional[Dict[str, np.ndarray]]:
    """Lee pp_a/pp_b de un norm_stats.npz (dict o NpzFile); None si no hay post-proceso."""
    if stats is None or any(k not in stats for k in PP_KEYS):
        return None
    return {k: np.asarray(stats[k], np.float32) for k in PP_KEYS}
