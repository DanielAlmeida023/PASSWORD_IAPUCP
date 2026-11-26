"""
INTI-3K · Gestor de descargas GFS (Open-Meteo) con caché de respaldo.

Reglas (Guía oficial + docstring de app/gfs_forecast.py):
  - No descargar en cada request: la grilla completa cuesta ~221 llamadas a Open-Meteo.
    Se reutiliza `gfs_fallback_cache.npz` mientras tenga menos de `refresh_h` horas (1–6 h, por
    defecto 3 h; variable de entorno INTI_GFS_REFRESH_H) y cubra el horizonte pedido.
  - Si Open-Meteo falla (sin internet, 429, 5xx) se usa la última caché, sin importar su edad,
    y la respuesta lo declara (`mode = "offline_cache_fallback"`) para que la UI lo muestre.
  - Lags ±24 h del modelo: siempre se pide `past_hours=24` y `forecast_hours = horizonte máx + 24`
    (+ margen = refresh máximo) para que cada hora mostrada tenga t-24 y t+24 reales.
  - Fechas pasadas: API histórica del MISMO GFS (`historical-forecast-api.open-meteo.com`,
    `models=gfs_global`, vía `get_gfs_forecast(start_date=...)`), con 1 día extra a cada lado para
    los lags; se guarda en disco de forma permanente (el pasado no cambia).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

MAX_HORIZON_H = 48          # horizonte máximo que ofrece la UI
LAG_FUTURE_H = 24           # lag +24 h del modelo (V7_LAGS)
LAG_PAST_H = 24             # lag -24 h del modelo
REFRESH_MIN_H, REFRESH_MAX_H = 1.0, 6.0
FETCH_FORECAST_H = MAX_HORIZON_H + LAG_FUTURE_H + int(REFRESH_MAX_H)  # 78 h hacia adelante
FAILURE_BACKOFF_S = 300     # tras un fallo de red no reintentar en cada request (5 min)
MAX_HIST_DAYS = 3


def _refresh_hours() -> float:
    try:
        v = float(os.environ.get("INTI_GFS_REFRESH_H", "3"))
    except ValueError:
        v = 3.0
    return float(min(max(v, REFRESH_MIN_H), REFRESH_MAX_H))


def utc_hour_str(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%d%H")


def select_live_output_indices(times, now_str: str, horizon_h: int) -> np.ndarray:
    """Horas a mostrar: desde la hora UTC actual, `horizon_h` horas (la secuencia trae ±24 h de contexto)."""
    t = np.asarray(times).astype(str)
    now_idx = int(np.searchsorted(t, now_str))
    now_idx = min(now_idx, max(0, len(t) - horizon_h))
    return np.arange(now_idx, min(now_idx + horizon_h, len(t)))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="minutes")


class GFSCacheManager:
    def __init__(self, cache_path: Path, hist_dir: Path, fetch_fn: Optional[Callable[..., Any]] = None):
        self.cache_path = Path(cache_path)
        self.hist_dir = Path(hist_dir)
        self._fetch_fn = fetch_fn
        self._lock = threading.Lock()
        self._last_failure_ts = 0.0
        self._last_error: Optional[str] = None

    # -- descarga -------------------------------------------------------------
    def _fetch(self, **kw):
        if self._fetch_fn is not None:
            return self._fetch_fn(**kw)
        from gfs_forecast import get_gfs_forecast  # app/gfs_forecast.py oficial

        return get_gfs_forecast(**kw)

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        if not self.cache_path.exists():
            return None
        try:
            d = np.load(self.cache_path)
            fetched_at = float(d["fetched_at"]) if "fetched_at" in d.files else self.cache_path.stat().st_mtime
            return {"X": d["X"].astype(np.float32), "times": d["times"].astype(str), "fetched_at": fetched_at}
        except Exception:
            return None

    def _save_cache(self, X: np.ndarray, times: np.ndarray, fetched_at: float) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, X=X.astype(np.float32), times=np.asarray(times).astype(str), fetched_at=fetched_at)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _covers(times: np.ndarray, needed_future_h: int, now_str: str) -> bool:
        t = np.asarray(times).astype(str)
        now_idx = int(np.searchsorted(t, now_str))
        return now_idx >= LAG_PAST_H and (len(t) - now_idx) >= needed_future_h + LAG_FUTURE_H

    def status(self) -> Dict[str, Any]:
        c = self._load_cache()
        return {
            "cache_file": self.cache_path.name,
            "cache_exists": c is not None,
            "cache_fetched_at_utc": _iso(c["fetched_at"]) if c else None,
            "cache_age_h": round((time.time() - c["fetched_at"]) / 3600.0, 2) if c else None,
            "refresh_every_h": _refresh_hours(),
            "last_error": self._last_error,
        }

    # -- pronóstico actual ----------------------------------------------------
    def get_live(self, horizon_h: int = 24, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Devuelve dict(X, times, mode, fetched_at, age_h, error). Lanza RuntimeError si no hay datos."""
        now_str = utc_hour_str(now)
        refresh_h = _refresh_hours()
        with self._lock:
            cache = self._load_cache()
            if cache is not None:
                age_h = (time.time() - cache["fetched_at"]) / 3600.0
                if age_h < refresh_h and self._covers(cache["times"], horizon_h, now_str):
                    return {**cache, "mode": "cache_fresh", "age_h": age_h, "error": None}

            error: Optional[str] = None
            in_backoff = (time.time() - self._last_failure_ts) < FAILURE_BACKOFF_S and cache is not None
            if not in_backoff:
                try:
                    X, times = self._fetch(forecast_hours=FETCH_FORECAST_H, past_hours=LAG_PAST_H)
                    X = np.asarray(X, dtype=np.float32)
                    times = np.asarray(times).astype(str)
                    if len(times) < LAG_PAST_H + horizon_h:
                        raise RuntimeError(f"Open-Meteo devolvió sólo {len(times)} horas completas")
                    fetched_at = time.time()
                    self._save_cache(X, times, fetched_at)
                    self._last_error = None
                    return {"X": X, "times": times, "fetched_at": fetched_at, "mode": "open_meteo_live",
                            "age_h": 0.0, "error": None}
                except Exception as exc:  # red caída, 429, JSON inválido...
                    self._last_failure_ts = time.time()
                    error = f"{type(exc).__name__}: {exc}"[:160]
                    self._last_error = error
            else:
                error = f"Reintento en pausa tras fallo reciente: {self._last_error}"

            if cache is None:
                raise RuntimeError(f"Open-Meteo no disponible y no hay caché local ({error})")
            age_h = (time.time() - cache["fetched_at"]) / 3600.0
            return {**cache, "mode": "offline_cache_fallback", "age_h": age_h, "error": error}

    # -- fechas pasadas ------------------------------------------------------
    @staticmethod
    def parse_hist_range(start_date: str, end_date: Optional[str]) -> tuple[date, date]:
        try:
            d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
            d1 = datetime.strptime(end_date or start_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("Fechas en formato YYYY-MM-DD") from exc
        if d1 < d0:
            raise ValueError("end_date debe ser >= start_date")
        if (d1 - d0).days + 1 > MAX_HIST_DAYS:
            raise ValueError(f"Rango máximo {MAX_HIST_DAYS} días")
        if d1 >= datetime.now(timezone.utc).date() - timedelta(days=1):
            raise ValueError("Para fechas recientes usa el pronóstico actual (fechas pasadas: hasta anteayer)")
        return d0, d1

    def get_historical(self, start_date: str, end_date: Optional[str]) -> Dict[str, Any]:
        """
        Fechas en hora Perú. Descarga [start-1 día, end+2 días] en UTC para cubrir t-24 y t+24 de
        todas las horas locales del rango (día local D = UTC D 05:00 .. D+1 04:00).
        """
        d0, d1 = self.parse_hist_range(start_date, end_date)
        f0, f1 = d0 - timedelta(days=1), d1 + timedelta(days=2)
        path = self.hist_dir / f"gfs_hist_{f0:%Y%m%d}_{f1:%Y%m%d}.npz"
        with self._lock:
            if path.exists():
                d = np.load(path)
                X, times, mode = d["X"].astype(np.float32), d["times"].astype(str), "historical_cache"
            else:
                X, times = self._fetch(start_date=f0.isoformat(), end_date=f1.isoformat())
                X, times = np.asarray(X, dtype=np.float32), np.asarray(times).astype(str)
                self.hist_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, X=X, times=times)
                mode = "open_meteo_historical"
        # horas de salida = horas locales (UTC-5) dentro de [d0, d1]
        lo = (datetime.combine(d0, datetime.min.time()) + timedelta(hours=5)).strftime("%Y%m%d%H")
        hi = (datetime.combine(d1 + timedelta(days=1), datetime.min.time()) + timedelta(hours=4)).strftime("%Y%m%d%H")
        out_idx = np.where((times >= lo) & (times <= hi))[0]
        return {"X": X, "times": times, "out_idx": out_idx, "mode": mode, "file": path.name}
