"""
Paridad de preprocesamiento producto vs. entrenamiento (lags ±24 h, relleno de bordes, M24/S24,
gradientes, punto de rocío y normalización), con GFS sintético. No requiere data de Kaggle ni red.

    cd product_app/backend && python -m pytest tests -q
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "product_app" / "backend" / "app", ROOT / "kaggle_pipeline" / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inti_inference as ii  # noqa: E402
import train_v7  # noqa: E402
from gfs_cache_manager import (  # noqa: E402
    FETCH_FORECAST_H,
    LAG_FUTURE_H,
    LAG_PAST_H,
    GFSCacheManager,
    select_live_output_indices,
)

LAGS = np.array(train_v7.V7_LAGS)


def _times(start: datetime, n: int) -> np.ndarray:
    return np.array([(start + timedelta(hours=k)).strftime("%Y%m%d%H") for k in range(n)])


def _synthetic_gfs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.empty((n, 17, 13, 4), dtype=np.float32)
    X[..., 0] = rng.uniform(30, 100, (n, 17, 13))    # RH %
    X[..., 1] = rng.uniform(265, 300, (n, 17, 13))   # T K
    X[..., 2:] = rng.normal(0, 3, (n, 17, 13, 2))    # U, V
    return X


def _hours(t):
    return np.array([np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}") for s in t], dtype="datetime64[h]")


def test_product_uses_training_feature_function():
    # El producto no reimplementa los lags: usa literalmente la función de entrenamiento.
    assert ii.build_v7_temporal_and_spatial_features is train_v7.build_v7_temporal_and_spatial_features
    assert tuple(ii.V7_LAGS) == (-24, -6, -3, -1, 0, 1, 3, 6, 24)


def test_live_window_has_exact_lags_for_every_output_hour():
    """Con past_hours=24 y forecast_hours=horizonte+24 todas las horas mostradas tienen t-24..t+24 reales."""
    horizon = 48
    start = datetime(2026, 11, 25, 18)
    n = LAG_PAST_H + FETCH_FORECAST_H                    # lo que devuelve Open-Meteo
    t = _times(start, n)
    X = _synthetic_gfs(n)
    now_str = t[LAG_PAST_H + 5]                          # caché con 5 h de antigüedad
    out = select_live_output_indices(t, now_str, horizon)
    assert len(out) == horizon and t[out[0]] == now_str

    W, M24, S24, GRAD, TD = ii.build_features(X, t)
    for i in out:
        for j, lag in enumerate(LAGS):
            np.testing.assert_array_equal(W[i, j], X[i + lag])
        win = X[i - 12 : i + 13]
        np.testing.assert_allclose(M24[i], win.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(S24[i], win.std(axis=0), rtol=1e-5, atol=1e-5)
    rep = ii.lag_report(t, out)
    assert rep["n_hours_all_lags_exact"] == horizon
    assert rep["n_lag_slots_self_fill"] == 0 and rep["n_lag_slots_near_fill"] == 0


def test_edge_fill_matches_training_function_exactly():
    """Sin t+24 (o con huecos) el producto rellena EXACTAMENTE como el entrenamiento:
    busca ±1, ±2, ±3 h alrededor del lag y si no hay nada usa la propia hora t."""
    n = 40
    t = _times(datetime(2024, 7, 28, 0), n)
    keep = np.ones(n, dtype=bool)
    keep[[10, 11, 20]] = False                           # huecos como en datos reales
    t, X = t[keep], _synthetic_gfs(n, seed=1)[keep]
    W = train_v7.build_v7_temporal_and_spatial_features(X, t)[0]
    src = ii.lag_source_indices(t)
    for i in range(len(t)):
        for j in range(len(LAGS)):
            np.testing.assert_array_equal(W[i, j], X[src[i, j]])

    # Reimplementación independiente de la regla del entrenamiento para documentarla
    hrs = _hours(t).astype(np.int64)
    pos = {int(h): k for k, h in enumerate(hrs)}
    for i, h in enumerate(hrs):
        for j, lag in enumerate(LAGS):
            k = pos.get(int(h + lag))
            if k is None:
                for d in (1, -1, 2, -2, 3, -3):
                    k = pos.get(int(h + lag + d))
                    if k is not None:
                        break
            assert src[i, j] == (k if k is not None else i)

    # Última hora: no existe t+24 ni vecinos -> se usa X[t] (relleno del entrenamiento)
    assert src[-1, -1] == len(t) - 1
    assert ii.lag_report(t)["n_lag_slots_self_fill"] > 0


def test_old_backend_margin_was_padding_t_plus_24():
    """Regresión: el backend anterior pedía forecast_hours=horizonte+6 -> las últimas ~18 h
    usaban X[t] en lugar de t+24 (fuera de la distribución de entrenamiento)."""
    horizon = 24
    n = 24 + horizon + 6
    t = _times(datetime(2026, 11, 25, 0), n)
    rep = ii.lag_report(t, np.arange(24, 24 + horizon))
    assert rep["n_hours_all_lags_exact"] < horizon


def test_normalization_matches_prepare_split_tensors():
    """normalize_features == normalización de train_v7.prepare_split_tensors (mismas stats)."""
    n = 60
    t = _times(datetime(2024, 6, 1, 0), n)
    X = _synthetic_gfs(n, seed=2)
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(n, 70, 35, 4)).astype(np.float32)
    st = {
        "dem": rng.uniform(2000, 6000, (70, 35)).astype(np.float32),
        "wrf_lats": np.linspace(-10.4, -8.6, 70).astype(np.float32),
        "wrf_lons": np.linspace(-78.2, -77.2, 35).astype(np.float32),
        "gfs_lats": np.round(np.arange(-7.25, -11.25 - 1e-9, -0.25), 2).astype(np.float32),
        "gfs_lons": np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2).astype(np.float32),
    }
    trn = np.ones(n, dtype=bool)
    d = train_v7.prepare_split_tensors(X, Y, t, st, trn, trn, "cpu")

    W, M24, S24, GRAD, TD = ii.build_features(X, t)
    stats = {   # mismas fórmulas con las que el notebook guarda models/norm_stats.npz
        "mu_x": X.mean(axis=(0, 1, 2)).astype(np.float32),
        "sd_x": X.std(axis=(0, 1, 2)).astype(np.float32),
        "mu_td": float(TD.mean()),
        "sd_td": float(TD.std() + 1e-6),
        "mu_grad": GRAD.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32),
        "sd_grad": (GRAD.std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32),
    }
    f = ii.normalize_features(W, M24, S24, GRAD, TD, stats)
    for k in ("Wn", "Mn", "Sn", "Gn", "Tn"):
        np.testing.assert_allclose(f[k], d[k], rtol=1e-5, atol=1e-5, err_msg=k)


def test_live_fetch_requests_24h_past_and_horizon_plus_24(tmp_path):
    calls = []
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=24)

    def fake_fetch(**kw):
        calls.append(kw)
        n = kw["past_hours"] + kw["forecast_hours"]
        return _synthetic_gfs(n), _times(start.replace(tzinfo=None), n)

    m = GFSCacheManager(tmp_path / "gfs_fallback_cache.npz", tmp_path / "hist", fetch_fn=fake_fetch)
    g = m.get_live(horizon_h=48)
    assert calls[0]["past_hours"] == 24
    assert calls[0]["forecast_hours"] >= 48 + LAG_FUTURE_H
    out = select_live_output_indices(g["times"], datetime.now(timezone.utc).strftime("%Y%m%d%H"), 48)
    assert ii.lag_report(g["times"], out)["n_hours_all_lags_exact"] == 48


def test_historical_range_is_padded_for_lags(tmp_path):
    calls = []

    def fake_fetch(**kw):
        calls.append(kw)
        d0 = datetime.strptime(kw["start_date"], "%Y-%m-%d")
        d1 = datetime.strptime(kw["end_date"], "%Y-%m-%d")
        n = int((d1 - d0).total_seconds() // 3600) + 24
        return _synthetic_gfs(n), _times(d0, n)

    m = GFSCacheManager(tmp_path / "c.npz", tmp_path / "hist", fetch_fn=fake_fetch)
    g = m.get_historical("2024-07-29", "2024-07-29")
    assert calls[0] == {"start_date": "2024-07-28", "end_date": "2024-07-31"}
    t = g["times"][g["out_idx"]]
    assert t[0] == "2024072905" and t[-1] == "2024073004" and len(t) == 24   # día local completo
    assert ii.lag_report(g["times"], g["out_idx"])["n_hours_all_lags_exact"] == 24
    # segunda llamada: sale del disco, sin red
    m.get_historical("2024-07-29", "2024-07-29")
    assert len(calls) == 1
