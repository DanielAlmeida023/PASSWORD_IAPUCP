"""
Backend FastAPI para el Componente 2 (Producto que genere valor) — Hackathon IA PUCP 2026 (AIRE-ANDINO).
Conecta el pronóstico GFS actual (Open-Meteo vía app/gfs_forecast.py con caché de 2h)
con el modelo oficial de downscaling (models/modelo.pt) para generar mapas e indicadores a 3 km (70x35)
sobre la Cordillera Blanca y el Callejón de Huaylas.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT_DIR / "kaggle_pipeline" / "src"))

from gfs_forecast import get_gfs_forecast, make_windows  # noqa: E402
from ejemplo_producto import predecir, a_hora_lima  # noqa: E402

app = FastAPI(
    title="AIRE-ANDINO 3km · API de Inteligencia Climática Cordillera Blanca",
    description=(
        "Downscaling climático en tiempo real (GFS 27 km -> WRF 3 km) con el mismo modelo de Kaggle "
        "(`models/modelo.pt`) para toma de decisiones en Agricultura, Montañismo, Gestión Municipal y Turismo."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Puntos de referencia reales en la grilla 70x35 (Cordillera Blanca y Callejón de Huaylas)
ZONAS_REFERENCIA = [
    {"id": "huaraz_valle", "nombre": "Huaraz / Valle del Santa", "lat": -9.53, "lon": -77.53, "perfil": "Agricultura & Ciudad"},
    {"id": "caraz_cultivos", "nombre": "Caraz / Campos Agrícolas", "lat": -9.05, "lon": -77.81, "perfil": "Agricultura Exportación"},
    {"id": "huascaran_ruta", "nombre": "Nevado Huascarán / Campo Base", "lat": -9.12, "lon": -77.60, "perfil": "Andinismo Alta Montaña"},
    {"id": "pastoruri_turismo", "nombre": "Glaciar Pastoruri / Ruta Turística", "lat": -9.93, "lon": -77.19, "perfil": "Turismo & Conservación"},
    {"id": "chacas_vertiente", "nombre": "Chacas / Conchucos Oriental", "lat": -9.16, "lon": -77.36, "perfil": "Gestión Municipal"},
]


class ForecastRequest(BaseModel):
    forecast_hours: int = Field(default=24, ge=6, le=72, description="Horas de pronóstico GFS hacia adelante")
    zona_id: str = Field(default="huaraz_valle", description="ID de la zona o punto en la Cordillera Blanca")
    umbral_helada_c: float = Field(default=0.0, description="Umbral crítico de helada (°C)")
    umbral_viento_ms: float = Field(default=10.0, description="Umbral de viento fuerte en montaña (m/s)")


@app.get("/api/health")
def health_check():
    pesos_ok = (MODELS_DIR / "modelo.pt").exists()
    stats_ok = (MODELS_DIR / "norm_stats.npz").exists()
    meta: Dict[str, Any] = {}
    if stats_ok:
        st = np.load(MODELS_DIR / "norm_stats.npz")
        meta = {
            "exp_id": str(st["exp_id"]),
            "arch": str(st["arch"]),
            "val_skill_score": round(float(st["best_score"]), 4),
            "weights_path": "models/modelo.pt",
        }
    return {
        "status": "ok",
        "model_loaded": pesos_ok and stats_ok,
        "active_model": meta,
        "zonas_disponibles": ZONAS_REFERENCIA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/downscale-live")
def downscale_live_forecast(req: ForecastRequest):
    static_path = DATA_DIR / "static.npz"
    if not static_path.exists():
        raise HTTPException(status_code=503, detail="No se encontró data/static.npz.")

    static = dict(np.load(static_path))
    cache_file = str(APP_DIR / "gfs_cache.npz")

    # Descarga GFS actual de Open-Meteo (con caché de 2 horas para cuidar el límite de la API)
    X, times = get_gfs_forecast(forecast_hours=req.forecast_hours, cache_path=cache_file, cache_max_age_h=2.0)
    Xw, times = make_windows(X, times)

    # Ejecuta el modelo entrenado de Kaggle (models/modelo.pt)
    Y = predecir(Xw, static, times=times)  # (H, 70, 35, 4) -> RH, T(K), U, V

    T_c = Y[..., 1] - 273.15
    RH = np.clip(Y[..., 0], 0.0, 100.0)
    W_spd = np.sqrt(Y[..., 2] ** 2 + Y[..., 3] ** 2)

    # Ubicar píxel de 3 km más cercano a la zona solicitada
    zona = next((z for z in ZONAS_REFERENCIA if z["id"] == req.zona_id), ZONAS_REFERENCIA[0])
    wlat, wlon = static["wrf_lats"], static["wrf_lons"]
    r_idx = int(np.argmin(np.abs(wlat - zona["lat"])))
    c_idx = int(np.argmin(np.abs(wlon - zona["lon"])))
    elev_m = float(static["dem"][r_idx, c_idx])

    # Serie horaria en ese píxel de 3 km
    hourly_series: List[Dict[str, Any]] = []
    for t_idx, t_utc in enumerate(times):
        hourly_series.append(
            {
                "time_utc": str(t_utc),
                "time_lima": a_hora_lima(t_utc),
                "temp_c": round(float(T_c[t_idx, r_idx, c_idx]), 2),
                "rh_pct": round(float(RH[t_idx, r_idx, c_idx]), 1),
                "wind_ms": round(float(W_spd[t_idx, r_idx, c_idx]), 2),
                "u_ms": round(float(Y[t_idx, r_idx, c_idx, 2]), 2),
                "v_ms": round(float(Y[t_idx, r_idx, c_idx, 3]), 2),
            }
        )

    tmin_zona = float(T_c[:, r_idx, c_idx].min())
    tmax_zona = float(T_c[:, r_idx, c_idx].max())
    wmax_zona = float(W_spd[:, r_idx, c_idx].max())
    hora_tmin = a_hora_lima(times[int(T_c[:, r_idx, c_idx].argmin())])

    # Resumen regional de todo el dominio 70x35 (2,450 celdas de 3 km)
    helada_mask = T_c.min(axis=0) < req.umbral_helada_c
    pct_territorio_helada = round(float(100.0 * helada_mask.mean()), 1)

    if tmin_zona < req.umbral_helada_c or wmax_zona >= req.umbral_viento_ms:
        status = "ALERTA CRÍTICA"
        rec = (
            f"Riesgo detectado en {zona['nombre']} ({elev_m:.0f} m s.n.m.): "
            f"T mín de {tmin_zona:.1f} °C ({hora_tmin}) y ráfagas de {wmax_zona:.1f} m/s. "
            "Activar protección térmica de cultivos / restringir ascenso en ruta de alta montaña."
        )
    elif tmin_zona < req.umbral_helada_c + 3.0 or wmax_zona >= req.umbral_viento_ms * 0.75:
        status = "PRECAUCIÓN OPERATIVA"
        rec = (
            f"Condiciones de vigilancia en {zona['nombre']} ({elev_m:.0f} m s.n.m.): "
            f"T mín de {tmin_zona:.1f} °C ({hora_tmin}). Programar riego preventivo o ventana temprana."
        )
    else:
        status = "VENTANA SEGURA"
        rec = (
            f"Condiciones favorables en {zona['nombre']} ({elev_m:.0f} m s.n.m.): "
            f"T entre {tmin_zona:.1f} °C y {tmax_zona:.1f} °C, viento máx {wmax_zona:.1f} m/s."
        )

    meta_model = {"exp_id": "fallback", "val_skill_score": 0.40}
    if (MODELS_DIR / "norm_stats.npz").exists():
        st = np.load(MODELS_DIR / "norm_stats.npz")
        meta_model = {"exp_id": str(st["exp_id"]), "val_skill_score": round(float(st["best_score"]), 4)}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "periodo_lima": f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])}",
        "model_info": meta_model,
        "zona": {
            **zona,
            "grid_row": r_idx,
            "grid_col": c_idx,
            "lat_real_3km": round(float(wlat[r_idx]), 3),
            "lon_real_3km": round(float(wlon[c_idx]), 3),
            "elevacion_m": round(elev_m, 0),
        },
        "kpis_zona": {
            "tmin_c": round(tmin_zona, 2),
            "tmax_c": round(tmax_zona, 2),
            "hora_tmin_lima": hora_tmin,
            "viento_max_ms": round(wmax_zona, 2),
            "rh_media_pct": round(float(RH[:, r_idx, c_idx].mean()), 1),
        },
        "resumen_cordillera": {
            "total_celdas_3km": int(Y.shape[1] * Y.shape[2]),
            "pct_territorio_helada": pct_territorio_helada,
            "tmin_dominio_c": round(float(T_c.min()), 2),
            "tmax_dominio_c": round(float(T_c.max()), 2),
            "viento_max_dominio_ms": round(float(W_spd.max()), 2),
        },
        "actionable_status": status,
        "recommendation": rec,
        "serie_horaria_3km": hourly_series[:24],
    }
