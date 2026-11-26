"""Caché de respaldo de Open-Meteo (sin internet), política de refresco y exposición a helada."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "product_app" / "backend" / "app", ROOT / "kaggle_pipeline" / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gfs_forecast  # noqa: E402
from gfs_cache_manager import GFSCacheManager  # noqa: E402


def _seq(n=102):
    start = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0) - timedelta(hours=24)
    t = np.array([(start + timedelta(hours=k)).strftime("%Y%m%d%H") for k in range(n)])
    X = np.full((n, 17, 13, 4), 280.0, dtype=np.float32)
    return X, t


def test_refresh_policy_does_not_fetch_every_request(tmp_path):
    calls = []

    def fetch(**kw):
        calls.append(kw)
        return _seq()

    m = GFSCacheManager(tmp_path / "gfs_fallback_cache.npz", tmp_path / "h", fetch_fn=fetch)
    assert m.get_live(24)["mode"] == "open_meteo_live"
    assert m.get_live(24)["mode"] == "cache_fresh"
    assert m.get_live(48)["mode"] == "cache_fresh"
    assert len(calls) == 1


def test_offline_uses_fallback_cache_when_requests_fails(tmp_path, monkeypatch):
    path = tmp_path / "gfs_fallback_cache.npz"
    X, t = _seq()
    old = datetime.now(timezone.utc).timestamp() - 10 * 3600   # caché vencida (10 h)
    np.savez_compressed(path, X=X, times=t, fetched_at=old)

    def no_internet(*a, **k):
        raise requests.exceptions.ConnectionError("sin internet (simulado)")

    monkeypatch.setattr(gfs_forecast.requests, "get", no_internet)
    m = GFSCacheManager(path, tmp_path / "h")            # usa el gfs_forecast.py oficial
    g = m.get_live(24)
    assert g["mode"] == "offline_cache_fallback"
    assert "ConnectionError" in g["error"]
    assert g["age_h"] > 9
    np.testing.assert_array_equal(g["times"], t)
    # mientras dura la pausa tras el fallo no se vuelve a llamar a la red
    monkeypatch.setattr(gfs_forecast.requests, "get", lambda *a, **k: pytest.fail("no debía llamar a la red"))
    assert m.get_live(24)["mode"] == "offline_cache_fallback"


def test_offline_without_cache_raises(tmp_path):
    def fetch(**kw):
        raise requests.exceptions.ConnectionError("down")

    m = GFSCacheManager(tmp_path / "none.npz", tmp_path / "h", fetch_fn=fetch)
    with pytest.raises(RuntimeError):
        m.get_live(24)


def test_frost_exposure_counts_real_hours_below_zero():
    from main import frost_exposure

    T = np.array([   # 5 horas x 3 celdas (°C, predicción 3 km)
        [2.0, 1.0, 3.0],
        [0.5, -0.2, 2.0],
        [-1.0, -2.5, 1.0],
        [-0.1, -1.0, 0.5],
        [1.0, 0.0, 0.2],     # 0.0 no es < 0
    ])
    f = frost_exposure(T, 0.0)
    assert f["frost_hours"] == 3            # horas 1, 2, 3
    assert f["cells_with_frost"] == 2       # celdas 0 y 1
    assert f["frost_cell_hours"] == 5
    assert f["frost_hours_worst_cell"] == 3  # celda 1 (mín −2.5)
