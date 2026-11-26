"""
INTI-3K · Inferencia del modelo oficial de Kaggle (EnsembleVN / EnsembleV8) sobre una secuencia horaria de GFS.

Este módulo es el ÚNICO punto de entrada del producto al modelo. No duplica el preprocesamiento:
  - Lags ±24 h, M24/S24, gradientes y punto de rocío -> `train_v7.build_v7_temporal_and_spatial_features`
    (la MISMA función que usa el entrenamiento y que copia `train/submission_notebook.ipynb`).
  - Grilla de muestreo, rasgos topográficos, suavizado DEM -> `train.py`.
  - Arquitectura -> `ensemble_vn.EnsembleVN` o `train_v8.EnsembleV8` (carga estricta de `models/modelo.pt`).
La única línea "propia" es la normalización con las estadísticas guardadas en `models/norm_stats.npz`,
que replica literalmente `train_v7.prepare_split_tensors` (verificado en
`product_app/backend/tests/test_lags.py`).

Convención de entrada: X (N, 17, 13, 4) GFS horario en unidades físicas (RH %, T K, U, V m/s),
fila 0 = norte, y `times` (N,) 'YYYYMMDDHH' en UTC. Para que la hora t vea t-24 y t+24 reales,
la secuencia debe incluir 24 h antes y 24 h después de las horas que se quieren predecir.

Arquitecturas soportadas (clave `arch` de norm_stats.npz):
  - 'ensemble_vn' : ensamble final de N refits (`kaggle_pipeline/src/ensemble_vn.EnsembleVN`, exportado por
                    `export_ensemble.py`; config de miembros en `members_config`, pesos float16 -> .float()).
  - 'ensemble_v8' : EnsembleV8 de 4 miembros (envío histórico 0.49844), por compatibilidad.
Carpeta de pesos: `models/` o la variable de entorno INTI_MODELS_DIR (tests con modelos temporales).

Variante de lags: `models/norm_stats.npz` puede declarar `lag_set` ("default" | "past", ver
`train_v7.LAG_SETS`; lo escribe `run_cv.py --refit --lags ...`). Si no lo declara se usa "default"
(lags ±24 h, comportamiento histórico). Con "past" el modelo solo lee GFS de horas <= t.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
SRC_DIR = RAIZ / "kaggle_pipeline" / "src"
MODELS_DIR = RAIZ / "models"  # default; INTI_MODELS_DIR lo sobrescribe (ver models_dir())
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402  (código de entrenamiento, no duplicado)
    build_sampling_grid,
    build_topographic_features,
    extract_hour_utc,
    smooth_2d,
)
from train_v7 import V7_LAGS, build_v7_temporal_and_spatial_features, get_lag_config  # noqa: E402

_LOCK = threading.Lock()
_RUNNER: Optional[Dict[str, Any]] = None
SUPPORTED_ARCHS = ("ensemble_vn", "ensemble_v8")


def models_dir() -> Path:
    """Carpeta de modelo.pt/norm_stats.npz: INTI_MODELS_DIR si está definida, si no <repo>/models."""
    env = os.environ.get("INTI_MODELS_DIR")
    return Path(env) if env else MODELS_DIR


def reset_runner() -> None:
    """Olvida el modelo cargado (p. ej. tras cambiar INTI_MODELS_DIR)."""
    global _RUNNER
    with _LOCK:
        _RUNNER = None


# ---------------------------------------------------------------------------
# Preprocesamiento (idéntico al entrenamiento)
# ---------------------------------------------------------------------------
def lag_set_from_stats(stats: Optional[Dict[str, Any]]) -> str:
    """Variante de lags declarada en las stats guardadas con el modelo (ausente => 'default')."""
    if stats is None or "lag_set" not in stats:
        return "default"
    name = str(np.asarray(stats["lag_set"]))
    get_lag_config(name)  # valida
    return name


def configured_lag_set() -> str:
    """lag_set de models/norm_stats.npz (o del runner ya cargado); 'default' si no hay modelo/clave."""
    if _RUNNER is not None and _RUNNER["models_dir"] == str(models_dir()):
        return _RUNNER["lag_set"]
    stats_path = models_dir() / "norm_stats.npz"
    if not stats_path.exists():
        return "default"
    with np.load(stats_path) as z:
        return lag_set_from_stats({k: z[k] for k in z.files if k == "lag_set"})


def build_features(X: np.ndarray, times: Sequence[str], lag_set: Optional[str] = None):
    """W (N,9,17,13,4), M24, S24, GRAD (N,6,17,13), TD (N,1,17,13) con la función de entrenamiento."""
    return build_v7_temporal_and_spatial_features(
        np.asarray(X, dtype=np.float32), np.asarray(times).astype(str), lag_set=lag_set or configured_lag_set()
    )


def normalize_features(W, M24, S24, GRAD, TD, stats) -> Dict[str, np.ndarray]:
    """Normalización de `train_v7.prepare_split_tensors` con stats guardadas (la misma función que usa
    `export_ensemble.py` para la submission: `ensemble_vn.normalize_features`)."""
    from ensemble_vn import normalize_features as _norm

    return _norm(W, M24, S24, GRAD, TD, stats)


def lag_source_indices(times: Sequence[str], lag_set: Optional[str] = None) -> np.ndarray:
    """
    Índice de la hora GFS que la función de entrenamiento usa para cada (hora, lag) -> (N, 9).
    Se obtiene corriendo la MISMA función sobre un tensor que codifica el índice (no se reimplementa
    la lógica de relleno): exacto, vecino ±1..3 h, o la propia hora t si no hay nada cerca.
    """
    t = np.asarray(times).astype(str)
    n = len(t)
    off = 1000.0  # desplazamiento para que el punto de rocío (Magnus) no divida por cero
    X_idx = np.broadcast_to((np.arange(n, dtype=np.float64) + off)[:, None, None, None], (n, 2, 2, 4))
    W = build_v7_temporal_and_spatial_features(
        np.ascontiguousarray(X_idx, dtype=np.float32), t, lag_set=lag_set or configured_lag_set()
    )[0]
    return np.rint(W[:, :, 0, 0, 1].astype(np.float64) - off).astype(np.int64)


def lag_report(times: Sequence[str], out_idx: Optional[Sequence[int]] = None, lag_set: Optional[str] = None) -> Dict[str, Any]:
    """Resume cuántas horas de salida tienen todos sus lags exactos vs. rellenados (como en el train)."""
    t = np.asarray(times).astype(str)
    ts =np.array([np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}") for s in t], dtype="datetime64[h]")
    lag_set = lag_set or configured_lag_set()
    lag_tuple = tuple(get_lag_config(lag_set)["lags"])
    src = lag_source_indices(t, lag_set)
    idx = np.arange(len(t)) if out_idx is None else np.asarray(out_idx, dtype=np.int64)
    lags = np.array(lag_tuple, dtype=np.int64)
    exact = np.zeros((len(idx), len(lags)), dtype=bool)
    near = np.zeros_like(exact)
    selff = np.zeros_like(exact)
    for k, i in enumerate(idx):
        got = (ts[src[i]] - ts[i]).astype(np.int64)
        exact[k] = got == lags
        selff[k] = (src[i] == i) & (lags != 0)
        near[k] = ~exact[k] & ~selff[k]
    padded_hours = [str(t[i]) for k, i in enumerate(idx) if not exact[k].all()]
    return {
        "lag_set": lag_set,
        "lags_h": list(lag_tuple),
        "n_output_hours": int(len(idx)),
        "n_hours_all_lags_exact": int(exact.all(axis=1).sum()),
        "n_lag_slots_near_fill": int(near.sum()),
        "n_lag_slots_self_fill": int(selff.sum()),
        "padded_hours_utc": padded_hours[:48],
    }


# ---------------------------------------------------------------------------
# Modelo (singleton en memoria)
# ---------------------------------------------------------------------------
def _build_runner(static: Dict[str, np.ndarray]) -> Dict[str, Any]:
    import torch

    mdir = models_dir()
    pesos_path = mdir / "modelo.pt"
    stats_path = mdir / "norm_stats.npz"
    if not (pesos_path.exists() and stats_path.exists()):
        raise FileNotFoundError(f"Faltan modelo.pt o norm_stats.npz en {mdir}")
    with np.load(stats_path) as z:
        arch = str(z["arch"])
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"norm_stats.npz declara arch={arch!r}; el producto soporta {SUPPORTED_ARCHS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if arch == "ensemble_vn":
        from ensemble_vn import load_ensemble_vn

        model, stats = load_ensemble_vn(mdir, static, device)
    else:
        from train_v8 import EnsembleV8

        stats = dict(np.load(stats_path))
        dem = static["dem"].astype(np.float32)
        grid_t = build_sampling_grid(static, device)
        topo_t = torch.tensor(build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"]), device=device)
        dem_smooth = smooth_2d(dem, k=9)
        lapse_phys = (-0.0065 * (dem - dem_smooth)).astype(np.float32)
        sd_x = stats["sd_x"]
        model = EnsembleV8(
            grid=grid_t,
            topo_features=topo_t,
            lapse_delta_norm=torch.tensor(lapse_phys / float(sd_x[1]), dtype=torch.float32, device=device),
            dem_delta_km=torch.tensor(((dem - dem_smooth) / 1000.0).astype(np.float32), device=device),
            mu_x=torch.tensor(stats["mu_x"], dtype=torch.float32, device=device),
            sd_x=torch.tensor(sd_x, dtype=torch.float32, device=device),
            anom_sd=torch.tensor(stats["anom_sd"], dtype=torch.float32, device=device),
            blend_weights=stats["blend_weights"],
        ).to(device)
        model.load_state_dict(torch.load(pesos_path, map_location=device, weights_only=True), strict=True)
        model.eval()
    return {
        "model": model, "stats": stats, "device": device, "arch": arch, "static_id": id(static),
        "lag_set": lag_set_from_stats(stats), "models_dir": str(mdir), "mtime": pesos_path.stat().st_mtime_ns,
    }


def get_runner(static: Dict[str, np.ndarray]) -> Dict[str, Any]:
    global _RUNNER
    with _LOCK:
        mdir = models_dir()
        if (
            _RUNNER is None
            or _RUNNER["static_id"] != id(static)
            or _RUNNER["models_dir"] != str(mdir)
            or _RUNNER["mtime"] != (mdir / "modelo.pt").stat().st_mtime_ns
        ):
            _RUNNER = _build_runner(static)
        return _RUNNER


def predict_sequence(
    X: np.ndarray,
    times: Sequence[str],
    static: Dict[str, np.ndarray],
    out_idx: Optional[Sequence[int]] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Corre el modelo servido (EnsembleVN o EnsembleV8) sobre las horas `out_idx` de la secuencia (por defecto todas).
    Las features se calculan sobre la secuencia COMPLETA (para que t±24 existan) y luego se recortan.
    Devuelve Y (len(out_idx), 70, 35, 4) en unidades físicas (RH %, T K, U, V m/s), fila 0 = sur.
    """
    import torch

    t = np.asarray(times).astype(str)
    idx = np.arange(len(t)) if out_idx is None else np.asarray(out_idx, dtype=np.int64)
    runner = get_runner(static)
    W, M24, S24, GRAD, TD = build_features(X, t, lag_set=runner["lag_set"])
    feats = normalize_features(W[idx], M24[idx], S24[idx], GRAD[idx], TD[idx], runner["stats"])
    h = extract_hour_utc(t[idx])
    dev = runner["device"]
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            sl = slice(s, s + batch_size)
            y = runner["model"](
                torch.tensor(feats["Wn"][sl], device=dev),
                torch.tensor(feats["Mn"][sl], device=dev),
                torch.tensor(feats["Sn"][sl], device=dev),
                torch.tensor(feats["Gn"][sl], device=dev),
                torch.tensor(feats["Tn"][sl], device=dev),
                torch.tensor(h[sl], dtype=torch.long, device=dev),
            ).cpu().numpy()
            y[..., 0] = np.clip(y[..., 0], 0.0, 100.0)  # igual que predict_v7 / predict_v8
            preds.append(y)
    if not preds:
        return np.zeros((0, 70, 35, 4), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)
