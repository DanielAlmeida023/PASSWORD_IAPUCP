"""
INTI-3K · Backend FastAPI — Centro de Comando Microclimático 3 km (Cordillera Blanca).
Hackathon IA PUCP 2026 · Equipo PASSWORD (DSC PUCP × IA-PUCP).

Núcleo obligatorio:
  Pronóstico GFS actual (Open-Meteo vía app/gfs_forecast.py, caché 1–6 h + respaldo offline)
  -> Modelo oficial de Kaggle (`models/modelo.pt` · EnsembleV8, ver `models/model_card.json`)
     con el MISMO preprocesamiento del entrenamiento (`app/inti_inference.py` importa train_v7/train_v8)
  -> Mapas 70x35 (3 km) de RH (%), T (°C), U (m/s), V (m/s) + Riesgo de Helada e Incendio.

Todas las horas de la data son UTC; la API devuelve además la hora Perú (UTC−5) para la UI.

Endpoints:
  - /api/health              : Estado del modelo, caché GFS y dominio.
  - /api/model-info          : Ficha del modelo (models/model_card.json) + verificación de pesos.
  - /api/grid-forecast       : Grilla 70x35 (IA 3 km + GFS 27 km interpolado). Acepta start_date/end_date.
  - /api/historical-forecast : Igual que grid-forecast pero exige start_date (fechas pasadas, GFS histórico).
  - /api/historical-cases    : Casos históricos sugeridos (p. ej. helada de julio 2024).
  - /api/analyze-geometry    : Pin / Polígono / Ruta: KPIs, exposición a helada, comparador GFS vs INTI-3K.
  - /api/infrastructure, /api/fires-firms, /api/fire-plume, /api/briefing.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BACKEND_APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_APP_DIR.parent.parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = Path(os.environ.get("INTI_MODELS_DIR") or (ROOT_DIR / "models"))  # = app/inti_inference.models_dir()
BACKEND_DATA_DIR = BACKEND_APP_DIR.parent / "data"

for _p in (BACKEND_APP_DIR, ROOT_DIR / "kaggle_pipeline" / "src", APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS  # noqa: E402
from inti_inference import lag_report, predict_sequence  # noqa: E402
from gfs_cache_manager import (  # noqa: E402
    GFSCacheManager,
    MAX_HORIZON_H,
    select_live_output_indices,
    utc_hour_str,
)

TZ_LABEL = "hora Perú, UTC−5"
_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

app = FastAPI(
    title="INTI-3K · Centro de Comando Microclimático Cordillera Blanca",
    description=(
        "API de downscaling climático 27 km -> 3 km con el modelo oficial de Kaggle EnsembleV8 "
        "(`models/modelo.pt`) para agricultores, COER Áncash, municipios y SERNANP."
    ),
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GFS = GFSCacheManager(
    cache_path=BACKEND_DATA_DIR / "gfs_fallback_cache.npz",
    hist_dir=BACKEND_DATA_DIR / "gfs_hist",
)

_FORECAST_CACHE: Dict[str, Any] = {}
_FORECAST_LOCK = threading.Lock()
_STATIC_CACHE: Optional[Dict[str, np.ndarray]] = None
_FIRMS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_WEIGHTS_SHA: Dict[str, str] = {}

HISTORICAL_CASES = [
    {
        "id": "helada_2024_07_29",
        "label": "Helada 29 jul 2024",
        "start_date": "2024-07-29",
        "end_date": "2024-07-29",
        "nota": "Noche más fría de julio 2024 en el GFS sobre la Cordillera Blanca (puntos > 5000 m bajo −9 °C).",
    },
    {
        "id": "helada_2024_07_18",
        "label": "Helada 18 jul 2024",
        "start_date": "2024-07-18",
        "end_date": "2024-07-18",
        "nota": "Segunda noche más fría de julio 2024; mínima de 3.7 °C en el GFS sobre Caraz.",
    },
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def hora_peru(t_utc: str) -> str:
    """'YYYYMMDDHH' (UTC) -> 'mié 26/11 19:00' en hora Perú (UTC−5)."""
    dt = datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)
    return f"{_DIAS[dt.weekday()]} {dt:%d/%m %H}:00"


def get_static() -> Dict[str, np.ndarray]:
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return _STATIC_CACHE
    candidates = [DATA_DIR / "static.npz", BACKEND_DATA_DIR / "static.npz"]
    env_path = os.environ.get("INTI_STATIC_PATH", "").strip()
    if env_path:
        candidates.insert(0, Path(env_path))
    for candidate in candidates:
        if candidate.exists():
            _STATIC_CACHE = dict(np.load(candidate))
            return _STATIC_CACHE
    raise HTTPException(
        status_code=503,
        detail="No se encontró data/static.npz. Descarga la data oficial en data/ siguiendo el README.",
    )


def load_model_card() -> Dict[str, Any]:
    path = MODELS_DIR / "model_card.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _WEIGHTS_SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _WEIGHTS_SHA[key] = h.hexdigest()
    return _WEIGHTS_SHA[key]


def get_model_metadata() -> Dict[str, Any]:
    """Resumen del modelo para la UI, leído de models/model_card.json + norm_stats.npz (sin hardcodear)."""
    card = load_model_card()
    pesos_path = MODELS_DIR / "modelo.pt"
    stats_path = MODELS_DIR / "norm_stats.npz"
    loaded = pesos_path.exists() and stats_path.exists()
    val = card.get("validation", {})
    meta: Dict[str, Any] = {
        "loaded": loaded,
        "name": card.get("name", "modelo"),
        "arch": card.get("arch", "desconocido"),
        "arch_label": card.get("name", "modelo"),
        "exp_id": card.get("exp_id"),
        "commit": card.get("commit"),
        "val_skill_score": val.get("skill_score"),
        "val_fold": val.get("fold"),
        "kaggle_public_lb": card.get("kaggle", {}).get("public_lb"),
        "weights_path": "models/modelo.pt",
        "weights_mb": round(pesos_path.stat().st_size / 1e6, 2) if pesos_path.exists() else 0.0,
        "val_rmse": {k: round(float(v), 2) for k, v in val.get("rmse_per_var", {}).items()},
    }
    if loaded:
        st = np.load(stats_path)
        if "arch" in st.files:
            meta["arch"] = str(st["arch"])
        if "val_rmse_per_var" in st.files:
            meta["val_rmse"] = {k: round(float(v), 2) for k, v in zip(["RH", "T", "U", "V"], st["val_rmse_per_var"])}
        if meta["val_skill_score"] is None and "best_score" in st.files:
            meta["val_skill_score"] = round(float(st["best_score"]), 4)
    return meta


def compute_fire_risk_index(T_c: np.ndarray, RH: np.ndarray, W_ms: np.ndarray) -> np.ndarray:
    """
    Índice de Peligro de Incendio Altoandino [0, 100] basado en física de ignición/propagación:
    combina baja humedad relativa (RH < 45%), velocidad de viento (W_ms) y temperatura diurna (T_c).
    """
    dryness = np.clip((65.0 - RH) / 45.0, 0.0, 1.0)
    wind_factor = np.clip(W_ms / 12.0, 0.0, 1.0)
    temp_factor = np.clip((T_c - 4.0) / 18.0, 0.0, 1.0)
    idx = (0.50 * dryness + 0.35 * wind_factor + 0.15 * temp_factor) * 100.0
    return np.clip(idx, 0.0, 100.0).astype(np.float32)


def compute_wind_chill_c(T_c: float, W_ms: float) -> float:
    """Sensación térmica en montaña (fórmula estándar de Wind Chill si T <= 10 °C y viento >= 1.3 m/s)."""
    v_kmh = W_ms * 3.6
    if T_c <= 10.0 and v_kmh >= 4.8:
        wc = 13.12 + 0.6215 * T_c - 11.37 * (v_kmh**0.16) + 0.3965 * T_c * (v_kmh**0.16)
        return round(float(wc), 2)
    return round(float(T_c), 2)


def interpolar_gfs_a_wrf(Xt: np.ndarray, wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> np.ndarray:
    """Bilineal por coordenadas: (N, 17, 13, 4) GFS (fila 0 = norte) -> (N, 70, 35, 4) WRF (fila 0 = sur).
    Es la referencia "GFS 27 km" del comparador (lo que diría el GFS en tu celda sin el modelo)."""
    lat_g = GFS_LATS[::-1]
    Xs = Xt[:, ::-1]
    fi = np.interp(wrf_lats, lat_g, np.arange(len(lat_g)))
    fj = np.interp(wrf_lons, GFS_LONS, np.arange(len(GFS_LONS)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat_g) - 2)
    di = (fi - i0)[None, :, None, None]
    j0 = np.clip(np.floor(fj).astype(int), 0, len(GFS_LONS) - 2)
    dj = (fj - j0)[None, None, :, None]
    A = Xs[:, i0][:, :, j0]
    B = Xs[:, i0][:, :, j0 + 1]
    C = Xs[:, i0 + 1][:, :, j0]
    D = Xs[:, i0 + 1][:, :, j0 + 1]
    return ((A * (1 - dj) + B * dj) * (1 - di) + (C * (1 - dj) + D * dj) * di).astype(np.float32)


def frost_exposure(T_sel: np.ndarray, umbral_c: float = 0.0) -> Dict[str, int]:
    """
    Exposición a helada sobre las celdas 3 km seleccionadas. T_sel: (horas, n_celdas) en °C (predicción).
      - frost_hours       : horas en que AL MENOS una celda está bajo el umbral.
      - cells_with_frost  : celdas que pasan al menos una hora bajo el umbral.
      - frost_cell_hours  : suma de (celda, hora) bajo el umbral.
      - frost_hours_worst_cell : horas bajo el umbral en la celda más fría.
    """
    below = np.asarray(T_sel) < umbral_c
    worst = int(np.asarray(T_sel).min(axis=0).argmin()) if below.size else 0
    return {
        "frost_hours": int(below.any(axis=1).sum()),
        "cells_with_frost": int(below.any(axis=0).sum()),
        "frost_cell_hours": int(below.sum()),
        "frost_hours_worst_cell": int(below[:, worst].sum()) if below.size else 0,
    }


FIRE_ALTA, FIRE_MODERADA = 65.0, 45.0


def _cell_txt(c: Dict[str, Any]) -> str:
    return f"celda 3 km ({c['lat']:.3f}, {c['lon']:.3f}; {c['elevacion_m']:.0f} m)"


def classify_alerts(
    label: str,
    frost: Dict[str, Any],
    wind: Dict[str, Any],
    fire: Dict[str, Any],
    umbral_helada_c: float = 0.0,
    umbral_viento_ms: float = 10.0,
) -> Dict[str, Any]:
    """
    Semáforo por tipo de riesgo. Usuario principal: productor de papa que decide
    "¿protejo mi chacra esta noche por helada?".
      - El semáforo y la recomendación PRINCIPALES dependen SOLO de la helada (T 3 km predicha).
      - Incendio/sequedad y viento son alertas SECUNDARIAS, cada una con su propia recomendación,
        su celda 3 km y su valor predicho. Nunca generan una recomendación térmica.
    """
    t = float(frost["tmin_c"])
    fc = frost["cell"]
    cita = (f"[{_cell_txt(fc)}: INTI-3K {t:.1f} °C vs GFS 27 km {float(frost['gfs_c']):.1f} °C, "
            f"{frost['hora_lima']}]")
    if t < umbral_helada_c:
        status = "ALERTA CRÍTICA"
        recommendation = (
            f"Helada en {label}: {t:.1f} °C a las {frost['hora_lima']} ({TZ_LABEL}), "
            f"{frost['frost_hours']} h bajo {umbral_helada_c:.0f} °C en {frost['cells_with_frost']}/"
            f"{frost['n_cells']} celdas 3 km. {cita} Protege tu chacra esta noche: riego ligero al atardecer "
            "o cobertura (mantas/paja) antes de la hora más fría."
        )
    elif t < umbral_helada_c + 3.0:
        status = "PRECAUCIÓN OPERATIVA"
        recommendation = (
            f"Riesgo de helada cercano en {label}: mínima de {t:.1f} °C a las {frost['hora_lima']} ({TZ_LABEL}). "
            f"{cita} Ten lista la protección (riego/cobertura) y revisa el pronóstico al anochecer."
        )
    else:
        status = "VENTANA SEGURA"
        recommendation = (
            f"Sin helada prevista en {label}: mínima de {t:.1f} °C a las {frost['hora_lima']} ({TZ_LABEL}). "
            f"{cita} No hace falta proteger la chacra contra helada esta noche."
        )

    secondary: List[Dict[str, Any]] = []
    fmax = float(fire["fire_max"])
    if fmax >= FIRE_MODERADA:
        rc = fire["rh_cell"]
        secondary.append({
            "tipo": "incendio_sequedad",
            "nivel": "alta" if fmax >= FIRE_ALTA else "moderada",
            "titulo": "Peligro de incendio / aire muy seco",
            "valor": round(fmax, 0),
            "unidad": "/100",
            "hora_lima": fire["hora_lima"],
            "cell": fire["cell"],
            "texto": (
                f"Índice de incendio {fmax:.0f}/100 en tu {_cell_txt(fire['cell'])} a las {fire['hora_lima']}; "
                f"humedad mínima {float(fire['rh_min_pct']):.0f}% en la {_cell_txt(rc)} a las {fire['rh_hora_lima']}."
            ),
            "recomendacion": (
                "No hagas quemas agrícolas ni de rastrojos; avisa al COER/brigada ante humo y ten agua a mano."
                if fmax >= FIRE_ALTA else
                "Evita quemas agrícolas en las horas secas y de más viento."
            ),
        })
    wmax = float(wind["wmax_ms"])
    if wmax >= umbral_viento_ms * 0.75:
        secondary.append({
            "tipo": "viento",
            "nivel": "alta" if wmax >= umbral_viento_ms else "moderada",
            "titulo": "Viento fuerte",
            "valor": round(wmax, 1),
            "unidad": "m/s",
            "hora_lima": wind["hora_lima"],
            "cell": wind["cell"],
            "texto": (
                f"Ráfagas de {wmax:.1f} m/s en tu {_cell_txt(wind['cell'])} a las {wind['hora_lima']} "
                f"(sensación térmica {float(wind['wind_chill_c']):.1f} °C)."
            ),
            "recomendacion": "Asegura coberturas y plásticos; evita crestas expuestas y quemas con viento.",
        })
    return {"status": status, "recommendation": recommendation, "secondary": secondary}


def _gfs_status_payload(mode: str, fetched_at: Optional[float], age_h: Optional[float], error: Optional[str],
                        extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    labels = {
        "open_meteo_live": "GFS actual descargado ahora de Open-Meteo",
        "cache_fresh": "GFS actual desde caché local (se refresca cada pocas horas)",
        "offline_cache_fallback": "SIN CONEXIÓN a Open-Meteo: usando la última caché guardada",
        "open_meteo_historical": "GFS histórico descargado de Open-Meteo (historical-forecast-api)",
        "historical_cache": "GFS histórico desde caché local",
    }
    payload = {
        "mode": mode,
        "label": labels.get(mode, mode),
        "using_cache": mode in ("cache_fresh", "offline_cache_fallback", "historical_cache"),
        "is_fallback": mode == "offline_cache_fallback",
        "fetched_at_utc": datetime.fromtimestamp(fetched_at, timezone.utc).isoformat(timespec="minutes")
        if fetched_at else None,
        "fetched_at_peru": (datetime.fromtimestamp(fetched_at, timezone.utc) - timedelta(hours=5)).strftime("%d/%m %H:%M")
        if fetched_at else None,
        "age_h": round(age_h, 2) if age_h is not None else None,
        "error": error,
        "refresh_every_h": GFS.status()["refresh_every_h"],
    }
    if extra:
        payload.update(extra)
    return payload


def load_or_run_forecast(
    forecast_hours: int = 24,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GFS (actual o histórico) -> EnsembleV8 -> tensores 3 km cacheados en memoria.

    Ventana temporal (V7_LAGS = -24..+24 h): la secuencia GFS incluye 24 h antes y >= 24 h después
    de las horas mostradas; las features se calculan sobre la secuencia completa con la función de
    entrenamiento y luego se recortan a las horas de salida (ver app/inti_inference.py).
    """
    static = get_static()
    t_fetch0 = time.time()

    if start_date:
        try:
            g = GFS.get_historical(start_date, end_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"No se pudo obtener el GFS histórico: {exc}")
        X, times, out_idx = g["X"], g["times"], g["out_idx"]
        if len(out_idx) == 0:
            raise HTTPException(status_code=503, detail="El GFS histórico no cubre el rango pedido.")
        cache_key = f"hist_{g['file']}_{start_date}_{end_date}"
        gfs_status = _gfs_status_payload(g["mode"], None, None, None, {"file": g["file"]})
    else:
        try:
            g = GFS.get_live(horizon_h=forecast_hours)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        X, times = g["X"], g["times"]
        out_idx = select_live_output_indices(times, utc_hour_str(), forecast_hours)
        cache_key = f"live_{g['fetched_at']:.0f}_{int(out_idx[0]) if len(out_idx) else 0}_{forecast_hours}"
        gfs_status = _gfs_status_payload(g["mode"], g["fetched_at"], g["age_h"], g["error"])

    with _FORECAST_LOCK:
        cached = _FORECAST_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "gfs_status": gfs_status}

    fetch_ms = (time.time() - t_fetch0) * 1000.0
    t_inf0 = time.time()
    try:
        Y = predict_sequence(X, times, static, out_idx=out_idx)  # (H, 70, 35, 4): RH, T(K), U, V
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {exc}")
    inf_ms = (time.time() - t_inf0) * 1000.0
    lags = lag_report(times, out_idx)

    times_out = [str(times[i]) for i in out_idx]
    X_gfs_interp = interpolar_gfs_a_wrf(X[out_idx], static["wrf_lats"], static["wrf_lons"])

    T_c = (Y[..., 1] - 273.15).astype(np.float32)
    RH = np.clip(Y[..., 0], 0.0, 100.0).astype(np.float32)
    U = Y[..., 2].astype(np.float32)
    V = Y[..., 3].astype(np.float32)
    W_ms = np.sqrt(U**2 + V**2).astype(np.float32)

    bundle = {
        "computed_at": time.time(),
        "fetch_ms": round(fetch_ms, 1),
        "inference_ms": round(inf_ms, 1),
        "source_mode": gfs_status["mode"],
        "historical": bool(start_date),
        "lag_report": lags,
        "times_utc": times_out,
        "times_lima": [hora_peru(t) for t in times_out],
        "T_c": T_c,
        "RH": RH,
        "U": U,
        "V": V,
        "W_ms": W_ms,
        "fire_idx": compute_fire_risk_index(T_c, RH, W_ms),
        "T_c_gfs": (X_gfs_interp[..., 1] - 273.15).astype(np.float32),
        "RH_gfs": np.clip(X_gfs_interp[..., 0], 0.0, 100.0).astype(np.float32),
        "W_ms_gfs": np.sqrt(X_gfs_interp[..., 2] ** 2 + X_gfs_interp[..., 3] ** 2).astype(np.float32),
    }
    with _FORECAST_LOCK:
        if len(_FORECAST_CACHE) > 12:
            _FORECAST_CACHE.clear()
        _FORECAST_CACHE[cache_key] = bundle
    return {**bundle, "gfs_status": gfs_status}


def point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting estándar para saber si un píxel (lat, lon) cae dentro de un polígono dibujado."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class GeometryAnalyzeRequest(BaseModel):
    geometry_type: Literal["pin", "polygon", "route"] = "pin"
    coordinates: List[List[float]] = Field(
        ...,
        description="Lista de puntos [[lat, lon], ...]. 1 punto para pin, >=3 para polígono, >=2 para ruta.",
    )
    label: str = Field(default="Zona Seleccionada", description="Nombre personalizado del área o ruta")
    forecast_hours: int = Field(default=24, ge=6, le=MAX_HORIZON_H)
    start_date: Optional[str] = Field(default=None, description="Fecha pasada opcional YYYY-MM-DD (hora Perú)")
    end_date: Optional[str] = Field(default=None, description="Fecha pasada fin YYYY-MM-DD (hora Perú)")
    umbral_helada_c: float = Field(default=0.0)
    umbral_viento_ms: float = Field(default=10.0)


class FirePlumeRequest(BaseModel):
    lat: float = Field(..., description="Latitud del foco de incendio")
    lon: float = Field(..., description="Longitud del foco de incendio")
    hour_idx: int = Field(default=0, ge=0, le=71, description="Índice horario de inicio")
    duration_hours: int = Field(default=8, ge=2, le=18, description="Horas de propagación proyectada")
    forecast_hours: int = Field(default=24, ge=6, le=MAX_HORIZON_H)
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)


class BriefingRequest(BaseModel):
    label: str
    geometry_type: str
    elevacion_min_m: float
    elevacion_max_m: float
    tmin_c: float
    tmax_c: float
    hora_tmin_lima: str
    viento_max_ms: float
    wind_chill_min_c: float
    rh_min_pct: float
    fire_risk_max: float
    actionable_status: str
    area_km2: Optional[float] = None
    comparador_texto: Optional[str] = None
    use_ollama_if_available: bool = True


# ---------------------------------------------------------------------------
# Endpoints REST
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    static = get_static()
    meta = get_model_metadata()
    return {
        "status": "ok",
        "product": "INTI-3K · Centro de Comando Microclimático Cordillera Blanca",
        "team": "PASSWORD",
        "model": meta,
        "gfs_cache": GFS.status(),
        "domain_bounds": {
            "lat_min": float(static["wrf_lats"].min()),
            "lat_max": float(static["wrf_lats"].max()),
            "lon_min": float(static["wrf_lons"].min()),
            "lon_max": float(static["wrf_lons"].max()),
            "grid_shape": [int(static["dem"].shape[0]), int(static["dem"].shape[1])],
            "dem_min_m": round(float(static["dem"].min()), 1),
            "dem_max_m": round(float(static["dem"].max()), 1),
        },
        "optional_integrations": {
            "nasa_firms_key_configured": bool(os.environ.get("FIRMS_MAP_KEY", "").strip()),
            "ollama_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_peru": (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%d/%m/%Y %H:%M"),
        "tz_label": TZ_LABEL,
    }


@app.get("/api/model-info")
def model_info():
    """Ficha del modelo leída de models/model_card.json + verificación en vivo de los pesos servidos."""
    card = load_model_card()
    if not card:
        raise HTTPException(status_code=404, detail="No se encontró models/model_card.json")
    pesos_path = MODELS_DIR / "modelo.pt"
    served: Dict[str, Any] = {"exists": pesos_path.exists()}
    if pesos_path.exists():
        sha = _sha256(pesos_path)
        served.update(
            size_bytes=pesos_path.stat().st_size,
            size_mb=round(pesos_path.stat().st_size / 1e6, 2),
            sha256=sha,
            matches_card=sha == card.get("weights", {}).get("sha256"),
        )
    return {**card, "served_weights": served, "summary": get_model_metadata()}


@app.get("/api/historical-cases")
def historical_cases():
    return {"cases": HISTORICAL_CASES, "tz_label": TZ_LABEL}


def _build_grid_payload(b: Dict[str, Any]) -> Dict[str, Any]:
    static = get_static()
    T_c, RH, U, V, W_ms, fire_idx = b["T_c"], b["RH"], b["U"], b["V"], b["W_ms"], b["fire_idx"]
    timeline_summary = []
    for i in range(len(b["times_utc"])):
        timeline_summary.append(
            {
                "idx": i,
                "time_utc": b["times_utc"][i],
                "time_lima": b["times_lima"][i],
                "tmin_c": round(float(T_c[i].min()), 1),
                "tmax_c": round(float(T_c[i].max()), 1),
                "wmax_ms": round(float(W_ms[i].max()), 1),
                "frost_pct": round(float((T_c[i] < 0.0).mean() * 100.0), 1),
                "high_fire_pct": round(float((fire_idx[i] >= 55.0).mean() * 100.0), 1),
            }
        )
    return {
        "model": get_model_metadata(),
        "source_mode": b["source_mode"],
        "gfs_status": b["gfs_status"],
        "historical": b["historical"],
        "lag_report": b["lag_report"],
        "inference_ms": b["inference_ms"],
        "tz_label": TZ_LABEL,
        "periodo_lima": f"{b['times_lima'][0]} → {b['times_lima'][-1]} ({TZ_LABEL})",
        "wrf_lats": np.round(static["wrf_lats"], 4).tolist(),
        "wrf_lons": np.round(static["wrf_lons"], 4).tolist(),
        "dem": np.round(static["dem"], 0).astype(int).tolist(),
        "timeline": timeline_summary,
        "frames": {
            "T_c": np.round(T_c, 1).tolist(),
            "RH": np.round(RH, 0).astype(int).tolist(),
            "U": np.round(U, 1).tolist(),
            "V": np.round(V, 1).tolist(),
            "W_ms": np.round(W_ms, 1).tolist(),
            "fire_idx": np.round(fire_idx, 0).astype(int).tolist(),
            "T_c_gfs": np.round(b["T_c_gfs"], 1).tolist(),
            "W_ms_gfs": np.round(b["W_ms_gfs"], 1).tolist(),
        },
        "domain_kpis": {
            "total_cells_3km": int(T_c.shape[1] * T_c.shape[2]),
            "pct_helada_periodo": round(float((T_c.min(axis=0) < 0.0).mean() * 100.0), 1),
            "pct_helada_gfs_periodo": round(float((b["T_c_gfs"].min(axis=0) < 0.0).mean() * 100.0), 1),
            "pct_riesgo_incendio_alto": round(float((fire_idx.max(axis=0) >= 55.0).mean() * 100.0), 1),
            "tmin_dominio_c": round(float(T_c.min()), 1),
            "tmax_dominio_c": round(float(T_c.max()), 1),
            "wmax_dominio_ms": round(float(W_ms.max()), 1),
            "max_gfs_vs_ai_diff_c": round(float(np.abs(T_c - b["T_c_gfs"]).max()), 1),
        },
    }


@app.get("/api/grid-forecast")
def get_grid_forecast(
    hours: int = Query(default=24, ge=6, le=MAX_HORIZON_H),
    start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD (hora Perú) para fechas pasadas"),
    end_date: Optional[str] = Query(default=None),
):
    """
    Grilla completa 70x35 (3 km) del modelo para las horas pedidas (GFS actual por defecto, o fechas
    pasadas con start_date/end_date), junto con el GFS 27 km interpolado para el comparador.
    """
    b = load_or_run_forecast(forecast_hours=hours, start_date=start_date, end_date=end_date)
    return _build_grid_payload(b)


@app.get("/api/historical-forecast")
def get_historical_forecast(
    start_date: str = Query(..., description="YYYY-MM-DD (hora Perú)"),
    end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD (máx. 3 días)"),
):
    """Fechas pasadas: GFS histórico de Open-Meteo (gfs_global) -> EnsembleV8 -> grilla 3 km."""
    b = load_or_run_forecast(start_date=start_date, end_date=end_date or start_date)
    return _build_grid_payload(b)


def _cell_info(static: Dict[str, np.ndarray], r: int, c: int) -> Dict[str, Any]:
    return {
        "row": int(r),
        "col": int(c),
        "lat": round(float(static["wrf_lats"][r]), 4),
        "lon": round(float(static["wrf_lons"][c]), 4),
        "elevacion_m": round(float(static["dem"][r, c]), 0),
    }


@app.post("/api/analyze-geometry")
def analyze_geometry(req: GeometryAnalyzeRequest):
    """
    Evalúa el microclima a 3 km para cualquier geometría dibujada por el usuario:
      - Pin: 1 celda exacta de 3 km.
      - Polígono: agrega todas las celdas 3 km dentro del polígono (área km², fracción bajo 0 °C, etc.).
      - Ruta (A->B): perfil tramo por tramo con elevación DEM, Wind Chill y ventanas seguras.
    Cada alerta cita la celda 3 km crítica y su valor predicho, junto al valor GFS 27 km interpolado.
    """
    if not req.coordinates:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un par [lat, lon].")

    static = get_static()
    b = load_or_run_forecast(
        forecast_hours=req.forecast_hours,
        start_date=req.start_date,
        end_date=req.end_date,
    )
    wlat, wlon, dem = static["wrf_lats"], static["wrf_lons"], static["dem"]
    T_c, RH, U, V, W_ms, fire_idx = b["T_c"], b["RH"], b["U"], b["V"], b["W_ms"], b["fire_idx"]
    T_c_gfs = b["T_c_gfs"]

    selected_cells: List[Tuple[int, int]] = []
    route_profile: List[Dict[str, Any]] = []
    total_distance_km = 0.0

    if req.geometry_type == "pin" or len(req.coordinates) == 1:
        lat, lon = req.coordinates[0]
        r = int(np.argmin(np.abs(wlat - lat)))
        c = int(np.argmin(np.abs(wlon - lon)))
        selected_cells = [(r, c)]

    elif req.geometry_type == "polygon" and len(req.coordinates) >= 3:
        poly = [(float(p[0]), float(p[1])) for p in req.coordinates]
        for r in range(len(wlat)):
            for c in range(len(wlon)):
                if point_in_polygon(float(wlat[r]), float(wlon[c]), poly):
                    selected_cells.append((r, c))
        if not selected_cells:
            c_lat = float(np.mean([p[0] for p in poly]))
            c_lon = float(np.mean([p[1] for p in poly]))
            selected_cells = [(int(np.argmin(np.abs(wlat - c_lat))), int(np.argmin(np.abs(wlon - c_lon))))]

    elif req.geometry_type == "route" and len(req.coordinates) >= 2:
        pts = [(float(p[0]), float(p[1])) for p in req.coordinates]
        seen = set()
        cum_km = 0.0
        for idx in range(len(pts) - 1):
            lat1, lon1 = pts[idx]
            lat2, lon2 = pts[idx + 1]
            seg_km = haversine_km(lat1, lon1, lat2, lon2)
            steps = max(2, int(math.ceil(seg_km / 1.5)))
            for s in range(steps + 1):
                if idx > 0 and s == 0:
                    continue
                frac = s / steps
                la = lat1 + frac * (lat2 - lat1)
                lo = lon1 + frac * (lon2 - lon1)
                r = int(np.argmin(np.abs(wlat - la)))
                c = int(np.argmin(np.abs(wlon - lo)))
                if (r, c) not in seen:
                    seen.add((r, c))
                    selected_cells.append((r, c))
                dist_here = cum_km + frac * seg_km
                tmin_p = float(T_c[:, r, c].min())
                wmax_p = float(W_ms[:, r, c].max())
                route_profile.append(
                    {
                        "dist_km": round(dist_here, 1),
                        "lat": round(la, 3),
                        "lon": round(lo, 3),
                        "elevacion_m": round(float(dem[r, c]), 0),
                        "tmin_c": round(tmin_p, 1),
                        "wmax_ms": round(wmax_p, 1),
                        "wind_chill_min_c": compute_wind_chill_c(tmin_p, wmax_p),
                        "fire_risk_max": round(float(fire_idx[:, r, c].max()), 0),
                    }
                )
            cum_km += seg_km
        total_distance_km = round(cum_km, 1)

    if not selected_cells:
        raise HTTPException(status_code=400, detail="Geometría inválida: pin=1 punto, polígono>=3, ruta>=2.")

    rows = np.array([rc[0] for rc in selected_cells])
    cols = np.array([rc[1] for rc in selected_cells])

    elevs = dem[rows, cols]
    T_sel = T_c[:, rows, cols]
    T_gfs_sel = T_c_gfs[:, rows, cols]
    RH_sel = RH[:, rows, cols]
    W_sel = W_ms[:, rows, cols]
    U_sel = U[:, rows, cols]
    V_sel = V[:, rows, cols]
    F_sel = fire_idx[:, rows, cols]

    # Celda crítica = celda 3 km (y hora) con la temperatura predicha más baja dentro de la geometría
    h_crit, k_crit = np.unravel_index(int(T_sel.argmin()), T_sel.shape)
    r_crit, c_crit = int(rows[k_crit]), int(cols[k_crit])
    cell = _cell_info(static, r_crit, c_crit)
    hora_crit = b["times_lima"][h_crit]
    ai_crit = float(T_sel[h_crit, k_crit])
    gfs_crit = float(T_gfs_sel[h_crit, k_crit])
    gfs_cell_min = float(T_gfs_sel[:, k_crit].min())
    cell_txt = f"celda 3 km ({cell['lat']:.3f}, {cell['lon']:.3f}; {cell['elevacion_m']:.0f} m)"

    hourly_series = []
    wind_chills = []
    for i in range(len(b["times_utc"])):
        t_min_h = float(T_sel[i].min())
        w_max_h = float(W_sel[i].max())
        wc_h = compute_wind_chill_c(t_min_h, w_max_h)
        wind_chills.append(wc_h)
        hourly_series.append(
            {
                "idx": i,
                "time_utc": b["times_utc"][i],
                "time_lima": b["times_lima"][i],
                "temp_min_c": round(t_min_h, 1),
                "temp_mean_c": round(float(T_sel[i].mean()), 1),
                "temp_cell_c": round(float(T_sel[i, k_crit]), 1),
                "temp_gfs_cell_c": round(float(T_gfs_sel[i, k_crit]), 1),
                "temp_gfs_27km_c": round(float(T_gfs_sel[i].min()), 1),
                "diff_gfs_vs_ai_c": round(float(T_sel[i, k_crit] - T_gfs_sel[i, k_crit]), 1),
                "rh_min_pct": round(float(RH_sel[i].min()), 0),
                "wind_max_ms": round(w_max_h, 1),
                "u_mean_ms": round(float(U_sel[i].mean()), 1),
                "v_mean_ms": round(float(V_sel[i].mean()), 1),
                "wind_chill_c": wc_h,
                "fire_risk": round(float(F_sel[i].max()), 0),
            }
        )

    frost = frost_exposure(T_sel, req.umbral_helada_c)
    frost_gfs = frost_exposure(T_gfs_sel, req.umbral_helada_c)
    tmin_abs = ai_crit
    tmax_abs = float(T_sel.max())
    wmax_abs = float(W_sel.max())
    rh_min_abs = float(RH_sel.min())
    fire_max_abs = float(F_sel.max())
    wc_min_abs = float(min(wind_chills))
    hora_tmin = hora_crit
    area_km2 = round(len(selected_cells) * 9.0, 1)

    comparador_texto = (
        f"GFS 27 km decía {gfs_crit:.1f} °C, INTI-3K dice {ai_crit:.1f} °C en tu {cell_txt} "
        f"el {hora_crit} ({TZ_LABEL})."
    )
    if ai_crit < req.umbral_helada_c <= gfs_cell_min:
        comparador_veredicto = "El GFS 27 km NO veía helada en esta celda; INTI-3K sí la detecta."
    elif gfs_cell_min < req.umbral_helada_c <= ai_crit:
        comparador_veredicto = "El GFS 27 km marcaba helada; INTI-3K la descarta en esta celda."
    else:
        comparador_veredicto = f"Diferencia de resolución: {ai_crit - gfs_crit:+.1f} °C (INTI-3K − GFS)."

    # Contexto de cada riesgo con SU celda 3 km y hora (el semáforo principal es la helada)
    h_w, k_w = np.unravel_index(int(W_sel.argmax()), W_sel.shape)
    h_f, k_f = np.unravel_index(int(F_sel.argmax()), F_sel.shape)
    h_r, k_r = np.unravel_index(int(RH_sel.argmin()), RH_sel.shape)
    alerts = classify_alerts(
        label=req.label,
        umbral_helada_c=req.umbral_helada_c,
        umbral_viento_ms=req.umbral_viento_ms,
        frost={
            "tmin_c": tmin_abs, "tmax_c": tmax_abs, "gfs_c": gfs_crit, "hora_lima": hora_crit, "cell": cell,
            "frost_hours": frost["frost_hours"], "cells_with_frost": frost["cells_with_frost"],
            "n_cells": len(selected_cells),
        },
        wind={
            "wmax_ms": wmax_abs, "wind_chill_c": wc_min_abs, "hora_lima": b["times_lima"][h_w],
            "cell": _cell_info(static, int(rows[k_w]), int(cols[k_w])),
        },
        fire={
            "fire_max": fire_max_abs, "hora_lima": b["times_lima"][h_f],
            "cell": _cell_info(static, int(rows[k_f]), int(cols[k_f])),
            "rh_min_pct": rh_min_abs, "rh_hora_lima": b["times_lima"][h_r],
            "rh_cell": _cell_info(static, int(rows[k_r]), int(cols[k_r])),
        },
    )
    status, recommendation = alerts["status"], alerts["recommendation"]

    return {
        "label": req.label,
        "geometry_type": req.geometry_type,
        "tz_label": TZ_LABEL,
        "source_mode": b["source_mode"],
        "gfs_status": b["gfs_status"],
        "historical": b["historical"],
        "cells_3km_count": len(selected_cells),
        "area_km2": area_km2,
        "route_distance_km": total_distance_km if req.geometry_type == "route" else None,
        "elevacion_min_m": round(float(elevs.min()), 0),
        "elevacion_max_m": round(float(elevs.max()), 0),
        "elevacion_media_m": round(float(elevs.mean()), 0),
        "critical_cell": {**cell, "hora_lima": hora_crit, "time_utc": b["times_utc"][h_crit]},
        "comparador": {
            "cell": cell,
            "hora_lima": hora_crit,
            "inti3k_c": round(ai_crit, 1),
            "gfs_27km_c": round(gfs_crit, 1),
            "gfs_27km_min_celda_c": round(gfs_cell_min, 1),
            "delta_c": round(ai_crit - gfs_crit, 1),
            "texto": comparador_texto,
            "veredicto": comparador_veredicto,
            "metodo_gfs": "GFS 27 km interpolado bilinealmente a la celda 3 km (sin modelo)",
        },
        "kpis": {
            "tmin_c": round(tmin_abs, 1),
            "tmax_c": round(tmax_abs, 1),
            "hora_tmin_lima": hora_tmin,
            "frost_hours": frost["frost_hours"],
            "cells_with_frost": frost["cells_with_frost"],
            "frost_cell_hours": frost["frost_cell_hours"],
            "frost_hours_worst_cell": frost["frost_hours_worst_cell"],
            "frost_hours_gfs": frost_gfs["frost_hours"],
            "viento_max_ms": round(wmax_abs, 1),
            "wind_chill_min_c": round(wc_min_abs, 1),
            "rh_min_pct": round(rh_min_abs, 0),
            "fire_risk_max": round(fire_max_abs, 0),
            "gfs_27km_tmin_c": round(float(T_gfs_sel.min()), 1),
            "delta_resolucion_c": round(tmin_abs - float(T_gfs_sel.min()), 1),
        },
        "actionable_status": status,
        "alerta_principal_tipo": "helada",
        "recommendation": recommendation,
        "alertas_secundarias": alerts["secondary"],
        "hourly_series": hourly_series,
        "route_profile": route_profile[:40] if route_profile else [],
        "selected_cells_coords": [[round(float(wlat[r]), 4), round(float(wlon[c]), 4)] for r, c in selected_cells[:80]],
    }


@app.get("/api/infrastructure")
def get_infrastructure():
    geojson_path = BACKEND_DATA_DIR / "infraestructura_ancash.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="No se encontró infraestructura_ancash.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/fires-firms")
def get_fires_firms():
    """
    Consulta focos de incendio activos en Áncash vía NASA FIRMS si FIRMS_MAP_KEY está configurado en .env.
    En modo Keyless (o si no hay incendios activos hoy), devuelve escenarios históricos/simulables
    claramente etiquetados para evaluar conos de humo con el viento 3 km de nuestro modelo.
    """
    firms_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    now = time.time()

    if firms_key:
        if _FIRMS_CACHE["data"] is not None and (now - _FIRMS_CACHE["ts"]) < 1800:
            return _FIRMS_CACHE["data"]
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{firms_key}/VIIRS_SNPP_NRT/-79.0,-11.25,-76.0,-7.25/2"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and "latitude" in r.text:
                lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
                headers = lines[0].split(",")
                hotspots = []
                for ln in lines[1:31]:
                    vals = ln.split(",")
                    row = dict(zip(headers, vals))
                    hotspots.append(
                        {
                            "id": f"firms_{len(hotspots)+1}",
                            "nombre": f"Foco Satelital VIIRS ({row.get('acq_date', '')})",
                            "lat": float(row["latitude"]),
                            "lon": float(row["longitude"]),
                            "frp_mw": float(row.get("frp", 5.0)),
                            "confidence": row.get("confidence", "nominal"),
                            "source": "NASA_FIRMS_LIVE",
                        }
                    )
                payload = {
                    "mode": "NASA_FIRMS_LIVE",
                    "badge": "Telemetría Satelital NASA FIRMS (últimas 48h)",
                    "count": len(hotspots),
                    "hotspots": hotspots,
                }
                _FIRMS_CACHE["ts"] = now
                _FIRMS_CACHE["data"] = payload
                return payload
        except Exception:
            pass

    demo_hotspots = [
        {
            "id": "sim_huaraz_oeste",
            "nombre": "Escenario Simulado · Quebrada Llaca / Huaraz",
            "lat": -9.47,
            "lon": -77.48,
            "frp_mw": 18.5,
            "confidence": "simulado",
            "source": "SIMULACION_TEMPORADA_SECA",
        },
        {
            "id": "sim_chacas_pastizal",
            "nombre": "Escenario Simulado · Pastizales Chacas / Conchucos",
            "lat": -9.18,
            "lon": -77.35,
            "frp_mw": 14.2,
            "confidence": "simulado",
            "source": "SIMULACION_TEMPORADA_SECA",
        },
        {
            "id": "sim_caraz_ladera",
            "nombre": "Escenario Simulado · Ladera Huaylas / Caraz",
            "lat": -9.02,
            "lon": -77.77,
            "frp_mw": 11.8,
            "confidence": "simulado",
            "source": "SIMULACION_TEMPORADA_SECA",
        },
    ]
    return {
        "mode": "SIMULATION_KEYLESS",
        "badge": "Escenarios de Simulación (Agrega FIRMS_MAP_KEY en .env para NASA FIRMS en vivo)",
        "count": len(demo_hotspots),
        "hotspots": demo_hotspots,
    }


@app.post("/api/fire-plume")
def simulate_fire_plume(req: FirePlumeRequest):
    """
    Calcula un cono realista de propagación de humo/fuego durante `duration_hours` horas,
    advectando la trayectoria paso a paso usando los vectores de viento (U, V) a 3 km
    y la humedad relativa (RH) predichos por nuestro modelo (`models/modelo.pt`).
    """
    static = get_static()
    b = load_or_run_forecast(
        forecast_hours=req.forecast_hours,
        start_date=req.start_date,
        end_date=req.end_date,
    )
    wlat, wlon = static["wrf_lats"], static["wrf_lons"]
    U, V, RH, fire_idx = b["U"], b["V"], b["RH"], b["fire_idx"]

    n_hours = len(b["times_utc"])
    start_h = min(req.hour_idx, n_hours - 1)
    steps = min(req.duration_hours, n_hours - start_h)

    cur_lat, cur_lon = float(req.lat), float(req.lon)
    trajectory = []
    left_edge = []
    right_edge = []

    for step in range(steps):
        h_i = start_h + step
        r = int(np.argmin(np.abs(wlat - cur_lat)))
        c = int(np.argmin(np.abs(wlon - cur_lon)))
        u_ms = float(U[h_i, r, c])
        v_ms = float(V[h_i, r, c])
        w_spd = math.hypot(u_ms, v_ms)
        rh_val = float(RH[h_i, r, c])
        f_risk = float(fire_idx[h_i, r, c])

        trajectory.append(
            {
                "step_h": step,
                "time_lima": b["times_lima"][h_i],
                "lat": round(cur_lat, 4),
                "lon": round(cur_lon, 4),
                "u_ms": round(u_ms, 2),
                "v_ms": round(v_ms, 2),
                "wind_ms": round(w_spd, 2),
                "rh_pct": round(rh_val, 0),
                "fire_risk": round(f_risk, 0),
            }
        )

        dx_km = (u_ms * 3.6) * 0.35
        dy_km = (v_ms * 3.6) * 0.35
        dlat = dy_km / 111.32
        dlon = dx_km / (111.32 * math.cos(math.radians(cur_lat)))

        spread_deg = 0.012 * (step + 1)
        norm = math.hypot(dlat, dlon) + 1e-6
        perp_lat = -dlon / norm * spread_deg
        perp_lon = dlat / norm * spread_deg

        left_edge.append([round(cur_lon + perp_lon, 4), round(cur_lat + perp_lat, 4)])
        right_edge.append([round(cur_lon - perp_lon, 4), round(cur_lat - perp_lat, 4)])

        cur_lat = float(np.clip(cur_lat + dlat, float(wlat.min()), float(wlat.max())))
        cur_lon = float(np.clip(cur_lon + dlon, float(wlon.min()), float(wlon.max())))

    cone_coords = [[round(req.lon, 4), round(req.lat, 4)]] + left_edge + list(reversed(right_edge)) + [
        [round(req.lon, 4), round(req.lat, 4)]
    ]

    avg_risk = float(np.mean([p["fire_risk"] for p in trajectory]))
    max_wind = float(np.max([p["wind_ms"] for p in trajectory]))

    return {
        "origin": {"lat": req.lat, "lon": req.lon},
        "start_time_lima": b["times_lima"][start_h],
        "duration_hours": steps,
        "avg_fire_risk": round(avg_risk, 1),
        "max_wind_ms": round(max_wind, 1),
        "trajectory": trajectory,
        "cone_geojson": {
            "type": "Feature",
            "properties": {
                "avg_fire_risk": round(avg_risk, 1),
                "max_wind_ms": round(max_wind, 1),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [cone_coords],
            },
        },
    }


@app.post("/api/briefing")
def generate_bilingual_briefing(req: BriefingRequest):
    """
    Genera el boletín operativo accionable en Español y Quechua Áncash (100% determinístico y sin claves).
    Si Ollama está ejecutándose localmente (OLLAMA_URL), ofrece enriquecimiento opcional.
    """
    area_txt = f" ({req.area_km2:.0f} km² analizados)" if req.area_km2 else ""
    es_text = (
        f"BOLETÍN INTI-3K [{req.actionable_status}] — {req.label}{area_txt} "
        f"({req.elevacion_min_m:.0f}–{req.elevacion_max_m:.0f} m s.n.m.). "
        f"Pronóstico a 3 km: temperatura mínima de {req.tmin_c:.1f} °C ({req.hora_tmin_lima}, {TZ_LABEL}) "
        f"con sensación térmica de hasta {req.wind_chill_min_c:.1f} °C, ráfagas de {req.viento_max_ms:.1f} m/s "
        f"y humedad mínima de {req.rh_min_pct:.0f}%. "
    )
    if req.comparador_texto:
        es_text += req.comparador_texto + " "
    # Acción principal = helada (decisión del productor); incendio y viento como avisos adicionales
    if req.tmin_c < 0.0:
        es_text += "Helada: protege tu chacra esta noche (riego ligero al atardecer o cobertura antes de la hora más fría)."
    elif req.tmin_c < 3.0:
        es_text += "Helada posible: ten lista la protección y revisa el pronóstico al anochecer."
    else:
        es_text += "Sin helada prevista: no hace falta proteger la chacra contra helada esta noche."
    if req.fire_risk_max >= FIRE_ALTA:
        es_text += " Aviso aparte por aire seco: no hagas quemas agrícolas ni de rastrojos."
    if req.viento_max_ms >= 10.0:
        es_text += " Aviso aparte por viento: asegura coberturas y evita crestas expuestas."

    estado_qu = {
        "ALERTA CRÍTICA": "LLAKIPAQ WILLAKUY (SINCHI MANCHAKUY)",
        "PRECAUCIÓN OPERATIVA": "ALLI RIKAPAKUY (YACHAYWAN PURINA)",
        "VENTANA SEGURA": "ALLI PACHA (MANA MANCHAKUYPAQ)",
    }.get(req.actionable_status, "WILLAKUY")

    if req.tmin_c < 0.0:
        accion_qu = "Chakrakunata, uywakunatapis chullunkupita tapaykuy tuta manaraq chayamuptin."
    else:
        accion_qu = "Pachaqa allillam kaykan chakrapaq, purinapaqpis."
    if req.fire_risk_max >= FIRE_ALTA:
        accion_qu += " Ama qiwakunata rupachiychu; sinchi wayrawan nina miranmanmi."
    if req.viento_max_ms >= 10.0:
        accion_qu += " Sinchi wayra kaptin ama hatun hirka hananman wichaychu."

    qu_text = (
        f"INTI-3K WILLAKUY [{estado_qu}] — {req.label}. "
        f"Kay jirkakunacho ({req.elevacion_min_m:.0f}–{req.elevacion_max_m:.0f} m): "
        f"aswan alalaqqa {req.tmin_c:.1f} °C kanqa ({req.hora_tmin_lima}, hora Perú UTC−5), "
        f"wayraqa {req.viento_max_ms:.1f} m/s kallpawan pukamunqa. "
        f"{accion_qu}"
    )

    llm_source = "plantilla_operativa_local"
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    if req.use_ollama_if_available:
        try:
            prompt = (
                "Eres un meteorólogo del COER Áncash. Resume en máximo 2 oraciones claras y accionables "
                f"para un agricultor o guía de montaña en Huaraz este reporte de 3 km:\n{es_text}"
            )
            r = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=2.0,
            )
            if r.status_code == 200 and r.json().get("response"):
                es_text = r.json()["response"].strip()
                llm_source = f"ollama:{ollama_model}"
        except Exception:
            pass

    return {
        "spanish": es_text,
        "quechua_ancash": qu_text,
        "generator": llm_source,
    }
