"""
INTI-3K · Backend FastAPI — Centro de Comando Microclimático 3 km (Cordillera Blanca).
Hackathon IA PUCP 2026 · Equipo PASSWORD (DSC PUCP × IA-PUCP).

Núcleo obligatorio:
  Pronóstico GFS actual (Open-Meteo vía app/gfs_forecast.py con caché + respaldo offline)
  -> Modelo oficial entrenado para Kaggle (`models/modelo.pt` · `EnsembleV8` 4 familias *Full DirectRes*)
  -> Mapas 70x35 (3 km) de RH (%), T (°C), U (m/s), V (m/s) + Riesgo de Helada e Incendio.

Funcionalidades tácticas y de decisión:
  - /api/health            : Estado del modelo, métricas de validación (RMSE/Skill) y caché.
  - /api/grid-forecast     : Grilla completa 70x35 (IA 3 km + GFS 27 km interpolado para comparador).
  - /api/analyze-geometry  : Análisis sobre Pin, Polígono (parcela/distrito) o Ruta (A->B) dibujados.
  - /api/infrastructure    : GeoJSON de lagunas glaciares, presas, refugios y ciudades de Áncash.
  - /api/fires-firms       : Focos activos NASA FIRMS (opcional con FIRMS_MAP_KEY) o escenarios de prueba.
  - /api/fire-plume        : Cono de propagación de fuego y dispersión de humo advectado con viento 3 km (U,V).
  - /api/briefing          : Boletín operativo accionable en Español y Quechua Áncash (+ Ollama opcional).
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
BACKEND_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT_DIR / "kaggle_pipeline" / "src"))

from gfs_forecast import get_gfs_forecast, make_windows  # noqa: E402
from ejemplo_producto import predecir, a_hora_lima, _interpolar_a_wrf  # noqa: E402

app = FastAPI(
    title="INTI-3K · Centro de Comando Microclimático Cordillera Blanca",
    description=(
        "API de downscaling climático 27 km -> 3 km con EnsembleDownscaler (`models/modelo.pt`) "
        "para COER Áncash, Municipios, SERNANP y Comunidades Altoandinas."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caché en memoria de inferencia para respuestas < 50 ms al interactuar en el mapa 3D
_FORECAST_CACHE: Dict[str, Any] = {}
_STATIC_CACHE: Optional[Dict[str, np.ndarray]] = None
_FIRMS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def get_static() -> Dict[str, np.ndarray]:
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return _STATIC_CACHE
    for candidate in (DATA_DIR / "static.npz", BACKEND_DATA_DIR / "static.npz"):
        if candidate.exists():
            _STATIC_CACHE = dict(np.load(candidate))
            return _STATIC_CACHE
    raise HTTPException(
        status_code=503,
        detail="No se encontró data/static.npz. Descarga la data oficial en data/ siguiendo el README.",
    )


def get_model_metadata() -> Dict[str, Any]:
    pesos_path = MODELS_DIR / "modelo.pt"
    stats_path = MODELS_DIR / "norm_stats.npz"
    if not (pesos_path.exists() and stats_path.exists()):
        return {
            "loaded": False,
            "arch": "fallback",
            "arch_label": "Fallback bilineal",
            "val_skill_score": 0.0,
            "weights_path": "models/modelo.pt",
            "weights_mb": 0.0,
            "val_rmse": {"RH": 9.69, "T": 0.98, "U": 1.74, "V": 1.63},
        }
    st = np.load(stats_path)
    val_rmse = (
        {k: round(float(v), 2) for k, v in zip(["RH", "T", "U", "V"], st["val_rmse_per_var"])}
        if "val_rmse_per_var" in st
        else {"RH": 8.58, "T": 0.93, "U": 1.62, "V": 1.54}
    )
    arch_str = str(st["arch"]) if "arch" in st else "ensemble_v8"
    if arch_str == "ensemble_v8":
        arch_label = "EnsembleV8 Full-DirectRes (ConvNeXt + MultiScale + UNet + CompactFNO + TemporalAttn)"
    elif arch_str == "ensemble_v7":
        arch_label = "EnsembleV7 (PhysicsConvNeXt + DirectResConvNeXt + CrossAttnUNet + CompactFNO)"
    else:
        arch_label = "EnsembleDownscaler (PhysicsUNet + PhysicsConvNeXt + PhysicsFNO)"
    return {
        "loaded": True,
        "exp_id": str(st["exp_id"]) if "exp_id" in st else arch_str,
        "arch": arch_str,
        "arch_label": arch_label,
        "val_skill_score": round(float(st["best_score"]), 4) if "best_score" in st else 0.4071,
        "weights_path": "models/modelo.pt",
        "weights_mb": round(pesos_path.stat().st_size / 1e6, 2),
        "val_rmse": val_rmse,
    }


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


def load_or_run_forecast(
    forecast_hours: int = 24,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Obtiene el pronóstico GFS (actual o histórico) con caché en disco + respaldo offline,
    corre nuestro modelo oficial (`models/modelo.pt`) y cachea los tensores 3 km en memoria.
    """
    cache_key = f"{start_date}_{end_date}" if start_date else f"live_{forecast_hours}"
    now = time.time()
    cached = _FORECAST_CACHE.get(cache_key)
    if cached and (now - cached["computed_at"]) < 1800:  # 30 min en memoria
        return cached

    static = get_static()
    disk_cache = APP_DIR / "gfs_cache.npz"
    fallback_cache = BACKEND_DATA_DIR / "gfs_fallback_cache.npz"
    source_mode = "open_meteo_live" if not start_date else "open_meteo_historical"

    t_fetch0 = time.time()
    try:
        if start_date:
            X, times = get_gfs_forecast(start_date=start_date, end_date=end_date or start_date)
        else:
            # Pedimos past_hours=24 y margen +6h para alimentar los lags [-24..+6] de EnsembleV8
            fetch_h = min(max(forecast_hours + 6, 30), 78)
            X, times = get_gfs_forecast(
                forecast_hours=fetch_h,
                past_hours=24,
                cache_path=str(disk_cache),
                cache_max_age_h=2.0,
            )
            # Si había una caché previa más corta (ej. de past_hours=6), refrescar con past_hours=24
            if len(X) < forecast_hours + 20 and disk_cache.exists():
                disk_cache.unlink(missing_ok=True)
                X, times = get_gfs_forecast(
                    forecast_hours=fetch_h,
                    past_hours=24,
                    cache_path=str(disk_cache),
                    cache_max_age_h=2.0,
                )
            if disk_cache.exists():
                BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(fallback_cache, X=X, times=times)
    except Exception as exc:
        # Respaldo automático en caso de fallo de red / límite de Open-Meteo (Guía pág. 8)
        backup_path = disk_cache if disk_cache.exists() else fallback_cache
        if backup_path.exists():
            d = np.load(backup_path)
            X, times = d["X"], d["times"]
            source_mode = "offline_cache_fallback"
        else:
            raise HTTPException(status_code=503, detail=f"Error obteniendo GFS de Open-Meteo sin caché local: {exc}")

    Xw, times_w = make_windows(X, times)

    t_inf0 = time.time()
    Y_all = predecir(Xw, static, times=times_w)  # (H, 70, 35, 4) -> RH, T(K), U, V
    inf_ms = (time.time() - t_inf0) * 1000.0

    # Recortar exactamente a las `forecast_hours` futuras (saltando las 24h pasadas si fue live)
    if not start_date and len(Y_all) > forecast_hours:
        start_idx = min(24, max(0, len(Y_all) - forecast_hours))
        end_idx = min(start_idx + forecast_hours, len(Y_all))
        Y = Y_all[start_idx:end_idx]
        X_sub = X[start_idx:end_idx]
        times_out = times_w[start_idx:end_idx]
    else:
        Y = Y_all[:72]
        X_sub = X[:72]
        times_out = times_w[:72]

    # Interpolación bilineal simple del GFS crudo 27 km -> 70x35 SOLO para el comparador visual GFS vs IA 3 km
    X_gfs_interp = _interpolar_a_wrf(X_sub, static["wrf_lats"], static["wrf_lons"])

    T_c = (Y[..., 1] - 273.15).astype(np.float32)
    RH = np.clip(Y[..., 0], 0.0, 100.0).astype(np.float32)
    U = Y[..., 2].astype(np.float32)
    V = Y[..., 3].astype(np.float32)
    W_ms = np.sqrt(U**2 + V**2).astype(np.float32)
    fire_idx = compute_fire_risk_index(T_c, RH, W_ms)

    T_c_gfs = (X_gfs_interp[..., 1] - 273.15).astype(np.float32)
    RH_gfs = np.clip(X_gfs_interp[..., 0], 0.0, 100.0).astype(np.float32)
    W_ms_gfs = np.sqrt(X_gfs_interp[..., 2] ** 2 + X_gfs_interp[..., 3] ** 2).astype(np.float32)

    bundle = {
        "computed_at": now,
        "fetch_ms": round((time.time() - t_fetch0) * 1000.0, 1),
        "inference_ms": round(inf_ms, 1),
        "source_mode": source_mode,
        "times_utc": [str(t) for t in times_out],
        "times_lima": [a_hora_lima(t) for t in times_out],
        "T_c": T_c,
        "RH": RH,
        "U": U,
        "V": V,
        "W_ms": W_ms,
        "fire_idx": fire_idx,
        "T_c_gfs": T_c_gfs,
        "RH_gfs": RH_gfs,
        "W_ms_gfs": W_ms_gfs,
    }
    _FORECAST_CACHE[cache_key] = bundle
    return bundle


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
    forecast_hours: int = Field(default=24, ge=6, le=48)
    start_date: Optional[str] = Field(default=None, description="Fecha histórica opcional YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="Fecha histórica fin YYYY-MM-DD")
    umbral_helada_c: float = Field(default=0.0)
    umbral_viento_ms: float = Field(default=10.0)


class FirePlumeRequest(BaseModel):
    lat: float = Field(..., description="Latitud del foco de incendio")
    lon: float = Field(..., description="Longitud del foco de incendio")
    hour_idx: int = Field(default=0, ge=0, le=47, description="Índice horario de inicio")
    duration_hours: int = Field(default=8, ge=2, le=18, description="Horas de propagación proyectada")
    forecast_hours: int = Field(default=24, ge=6, le=48)
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
    }


@app.get("/api/grid-forecast")
def get_grid_forecast(
    hours: int = Query(default=24, ge=6, le=48),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
):
    """
    Devuelve la grilla completa 70x35 (3 km) de nuestro modelo para todas las horas solicitadas,
    junto con la versión GFS 27 km interpolada para el comparador visual instantáneo en el mapa 3D.
    """
    static = get_static()
    b = load_or_run_forecast(forecast_hours=hours, start_date=start_date, end_date=end_date)

    T_c = b["T_c"]
    RH = b["RH"]
    U = b["U"]
    V = b["V"]
    W_ms = b["W_ms"]
    fire_idx = b["fire_idx"]

    timeline_summary = []
    for i in range(len(b["times_utc"])):
        frost_pct = float((T_c[i] < 0.0).mean() * 100.0)
        high_fire_pct = float((fire_idx[i] >= 55.0).mean() * 100.0)
        timeline_summary.append(
            {
                "idx": i,
                "time_utc": b["times_utc"][i],
                "time_lima": b["times_lima"][i],
                "tmin_c": round(float(T_c[i].min()), 1),
                "tmax_c": round(float(T_c[i].max()), 1),
                "wmax_ms": round(float(W_ms[i].max()), 1),
                "frost_pct": round(frost_pct, 1),
                "high_fire_pct": round(high_fire_pct, 1),
            }
        )

    return {
        "model": get_model_metadata(),
        "source_mode": b["source_mode"],
        "inference_ms": b["inference_ms"],
        "periodo_lima": f"{b['times_lima'][0]} → {b['times_lima'][-1]}",
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
            "pct_helada_24h": round(float((T_c.min(axis=0) < 0.0).mean() * 100.0), 1),
            "pct_riesgo_incendio_alto": round(float((fire_idx.max(axis=0) >= 55.0).mean() * 100.0), 1),
            "tmin_dominio_c": round(float(T_c.min()), 1),
            "tmax_dominio_c": round(float(T_c.max()), 1),
            "wmax_dominio_ms": round(float(W_ms.max()), 1),
            "max_gfs_vs_ai_diff_c": round(float(np.abs(T_c - b["T_c_gfs"]).max()), 1),
        },
    }


@app.post("/api/analyze-geometry")
def analyze_geometry(req: GeometryAnalyzeRequest):
    """
    Evalúa el microclima a 3 km para cualquier geometría dibujada por el usuario:
      - Pin: 1 celda exacta de 3 km.
      - Polígono: agrega todas las celdas 3 km dentro del polígono (área km², fracción bajo 0 °C, etc.).
      - Ruta (A->B): perfil tramo por tramo con elevación DEM, Wind Chill y ventanas seguras.
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

    hourly_series = []
    frost_hours = 0
    wind_chills = []
    for i in range(len(b["times_utc"])):
        t_min_h = float(T_sel[i].min())
        t_mean_h = float(T_sel[i].mean())
        t_gfs_h = float(T_gfs_sel[i].mean())
        w_max_h = float(W_sel[i].max())
        rh_min_h = float(RH_sel[i].min())
        f_max_h = float(F_sel[i].max())
        wc_h = compute_wind_chill_c(t_min_h, w_max_h)
        wind_chills.append(wc_h)
        if t_min_h < req.umbral_helada_c:
            frost_hours += 1
        hourly_series.append(
            {
                "idx": i,
                "time_utc": b["times_utc"][i],
                "time_lima": b["times_lima"][i],
                "temp_min_c": round(t_min_h, 1),
                "temp_mean_c": round(t_mean_h, 1),
                "temp_gfs_27km_c": round(t_gfs_h, 1),
                "diff_gfs_vs_ai_c": round(t_mean_h - t_gfs_h, 1),
                "rh_min_pct": round(rh_min_h, 0),
                "wind_max_ms": round(w_max_h, 1),
                "u_mean_ms": round(float(U_sel[i].mean()), 1),
                "v_mean_ms": round(float(V_sel[i].mean()), 1),
                "wind_chill_c": wc_h,
                "fire_risk": round(f_max_h, 0),
            }
        )

    tmin_abs = float(T_sel.min())
    tmax_abs = float(T_sel.max())
    wmax_abs = float(W_sel.max())
    rh_min_abs = float(RH_sel.min())
    fire_max_abs = float(F_sel.max())
    wc_min_abs = float(min(wind_chills))
    t_min_hour_idx = int(T_sel.min(axis=1).argmin())
    hora_tmin = b["times_lima"][t_min_hour_idx]
    cells_with_frost = int((T_sel.min(axis=0) < req.umbral_helada_c).sum())
    area_km2 = round(len(selected_cells) * 9.0, 1)

    if tmin_abs < req.umbral_helada_c or wmax_abs >= req.umbral_viento_ms or fire_max_abs >= 65.0:
        status = "ALERTA CRÍTICA"
        triggers = []
        if tmin_abs < req.umbral_helada_c:
            triggers.append(f"helada de {tmin_abs:.1f} °C ({hora_tmin}, {frost_hours}h bajo 0 °C)")
        if wmax_abs >= req.umbral_viento_ms:
            triggers.append(f"ráfagas de {wmax_abs:.1f} m/s (sensación térmica {wc_min_abs:.1f} °C)")
        if fire_max_abs >= 65.0:
            triggers.append(f"peligro alto de incendio ({fire_max_abs:.0f}/100, RH mín {rh_min_abs:.0f}%)")
        recommendation = (
            f"Riesgo severo en {req.label}: " + "; ".join(triggers) + ". "
            "Activar protección térmica de cultivos (riego/cobertura), alertar brigadas COER y evitar ascensos en ventana crítica."
        )
    elif tmin_abs < req.umbral_helada_c + 3.0 or wmax_abs >= req.umbral_viento_ms * 0.75 or fire_max_abs >= 45.0:
        status = "PRECAUCIÓN OPERATIVA"
        recommendation = (
            f"Vigilancia activa en {req.label}: mínima de {tmin_abs:.1f} °C ({hora_tmin}), viento máx {wmax_abs:.1f} m/s "
            f"y riesgo de incendio {fire_max_abs:.0f}/100. Preparar medidas preventivas antes del anochecer."
        )
    else:
        status = "VENTANA SEGURA"
        recommendation = (
            f"Condiciones favorables en {req.label}: temperatura entre {tmin_abs:.1f} °C y {tmax_abs:.1f} °C, "
            f"viento máx {wmax_abs:.1f} m/s. Apto para labores agrícolas y tránsito en ruta."
        )

    return {
        "label": req.label,
        "geometry_type": req.geometry_type,
        "cells_3km_count": len(selected_cells),
        "area_km2": area_km2,
        "route_distance_km": total_distance_km if req.geometry_type == "route" else None,
        "elevacion_min_m": round(float(elevs.min()), 0),
        "elevacion_max_m": round(float(elevs.max()), 0),
        "elevacion_media_m": round(float(elevs.mean()), 0),
        "kpis": {
            "tmin_c": round(tmin_abs, 1),
            "tmax_c": round(tmax_abs, 1),
            "hora_tmin_lima": hora_tmin,
            "frost_hours": frost_hours,
            "cells_with_frost": cells_with_frost,
            "viento_max_ms": round(wmax_abs, 1),
            "wind_chill_min_c": round(wc_min_abs, 1),
            "rh_min_pct": round(rh_min_abs, 0),
            "fire_risk_max": round(fire_max_abs, 0),
            "gfs_27km_tmin_c": round(float(T_gfs_sel.min()), 1),
            "delta_resolucion_c": round(tmin_abs - float(T_gfs_sel.min()), 1),
        },
        "actionable_status": status,
        "recommendation": recommendation,
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
        f"Pronóstico a 3 km: temperatura mínima de {req.tmin_c:.1f} °C ({req.hora_tmin_lima}) "
        f"con sensación térmica de hasta {req.wind_chill_min_c:.1f} °C, ráfagas de {req.viento_max_ms:.1f} m/s "
        f"y humedad mínima de {req.rh_min_pct:.0f}%. "
    )
    if req.tmin_c <= 0.0:
        es_text += "Acción inmediata: activar protección contra heladas en cultivos y ganado antes de la medianoche."
    elif req.fire_risk_max >= 60.0:
        es_text += "Acción inmediata: prohibir quemas agrícolas y mantener brigadas en alerta por viento seco."
    elif req.viento_max_ms >= 10.0:
        es_text += "Acción inmediata: restringir ascenso en crestas expuestas por ráfagas fuertes."
    else:
        es_text += "Condiciones estables para labores agrícolas, turismo y operación municipal."

    estado_qu = {
        "ALERTA CRÍTICA": "LLAKIPAQ WILLAKUY (SINCHI MANCHAKUY)",
        "PRECAUCIÓN OPERATIVA": "ALLI RIKAPAKUY (YACHAYWAN PURINA)",
        "VENTANA SEGURA": "ALLI PACHA (MANA MANCHAKUYPAQ)",
    }.get(req.actionable_status, "WILLAKUY")

    if req.tmin_c <= 0.0:
        accion_qu = "Chakrakunata, uywakunatapis chullunkupita tapaykuy tuta manaraq chayamuptin."
    elif req.fire_risk_max >= 60.0:
        accion_qu = "Ama qiwakunata rupachiychu; sinchi wayrawan nina miranmanmi."
    elif req.viento_max_ms >= 10.0:
        accion_qu = "Sinchi wayra kaptin ama hatun hirka hananman wichaychu."
    else:
        accion_qu = "Pachaqa allillam kaykan chakrapaq, purinapaqpis."

    qu_text = (
        f"INTI-3K WILLAKUY [{estado_qu}] — {req.label}. "
        f"Kay jirkakunacho ({req.elevacion_min_m:.0f}–{req.elevacion_max_m:.0f} m): "
        f"aswan alalaqqa {req.tmin_c:.1f} °C kanqa ({req.hora_tmin_lima}), "
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
