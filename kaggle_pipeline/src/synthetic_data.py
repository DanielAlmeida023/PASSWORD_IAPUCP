"""
Datos sintéticos con las formas oficiales, para tests y `run_cv.py --smoke` (sin data de Kaggle).
X: (N, 17, 13, 4) GFS fila 0 = norte · Y: (N, 70, 35, 4) WRF fila 0 = sur · times 'YYYYMMDDHH'.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np


def hourly_times(start: str = "2024021800", end: str = "2024063023", step_h: int = 1) -> np.ndarray:
    t0 = datetime.strptime(start, "%Y%m%d%H")
    t1 = datetime.strptime(end, "%Y%m%d%H")
    n = int((t1 - t0).total_seconds() // 3600) // step_h + 1
    return np.array([(t0 + timedelta(hours=i * step_h)).strftime("%Y%m%d%H") for i in range(n)])


def make_static(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    wrf_lats = np.linspace(-10.5, -8.0, 70).astype(np.float32)  # fila 0 = sur (creciente)
    wrf_lons = np.linspace(-78.6, -77.4, 35).astype(np.float32)
    gfs_lats = np.linspace(-7.5, -11.0, 17).astype(np.float32)  # fila 0 = norte (decreciente)
    gfs_lons = np.linspace(-79.0, -77.0, 13).astype(np.float32)
    yy, xx = np.meshgrid(np.linspace(0, 1, 70), np.linspace(0, 1, 35), indexing="ij")
    dem = (500 + 4500 * np.exp(-((xx - 0.6) ** 2) / 0.05) + 100 * rng.standard_normal((70, 35))).astype(np.float32)
    return {"dem": dem, "wrf_lats": wrf_lats, "wrf_lons": wrf_lons, "gfs_lats": gfs_lats, "gfs_lons": gfs_lons}


def make_dataset(times: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """X, Y con ciclo diurno + ruido, valores físicamente plausibles (RH %, T K, U, V m/s)."""
    rng = np.random.default_rng(seed)
    n = len(times)
    h = np.array([int(t[-2:]) for t in times], dtype=np.float32)
    diurnal = np.sin(2 * np.pi * h / 24.0)[:, None, None]
    X = np.empty((n, 17, 13, 4), dtype=np.float32)
    X[..., 0] = 60 + 20 * diurnal + 5 * rng.standard_normal((n, 17, 13))
    X[..., 1] = 285 + 5 * diurnal + 1.0 * rng.standard_normal((n, 17, 13))
    X[..., 2] = 2 * diurnal + rng.standard_normal((n, 17, 13))
    X[..., 3] = -1 + rng.standard_normal((n, 17, 13))
    Y = np.empty((n, 70, 35, 4), dtype=np.float32)
    d2 = diurnal[:, :1, :1]
    Y[..., 0] = np.clip(60 + 20 * d2 + 8 * rng.standard_normal((n, 70, 35)), 0, 100)
    Y[..., 1] = 283 + 6 * d2 + 1.5 * rng.standard_normal((n, 70, 35))
    Y[..., 2] = 2 * d2 + 1.5 * rng.standard_normal((n, 70, 35))
    Y[..., 3] = -1 + 1.5 * rng.standard_normal((n, 70, 35))
    return X, Y
