"""Semáforo por tipo: la helada manda; incendio/sequedad y viento son secundarios con su propia acción."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "product_app" / "backend" / "app", ROOT / "kaggle_pipeline" / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main  # noqa: E402



def _fake_env(monkeypatch, T_c: float, RH: float, W: float):
    H, R, C = 12, 70, 35
    static = {
        "dem": np.full((R, C), 3000.0, dtype=np.float32),
        "wrf_lats": np.linspace(-10.4, -8.6, R).astype(np.float32),
        "wrf_lons": np.linspace(-78.2, -77.2, C).astype(np.float32),
    }
    T = np.full((H, R, C), T_c, dtype=np.float32)
    T[:, 40, 20] -= 0.5                                   # celda algo más fría
    RHa = np.full((H, R, C), 60.0, dtype=np.float32)
    RHa[5, 40, 21] = RH                                   # aire muy seco en una celda vecina
    U = np.full((H, R, C), W, dtype=np.float32)
    V = np.zeros_like(U)
    Wm = np.sqrt(U**2 + V**2)
    times = [f"20240729{h:02d}" for h in range(H)]
    bundle = {
        "source_mode": "historical_cache", "gfs_status": {"mode": "historical_cache"}, "historical": True,
        "times_utc": times, "times_lima": [main.hora_peru(t) for t in times],
        "T_c": T, "RH": RHa, "U": U, "V": V, "W_ms": Wm,
        "fire_idx": main.compute_fire_risk_index(T, RHa, Wm),
        "T_c_gfs": T + 2.0, "RH_gfs": RHa, "W_ms_gfs": Wm,
    }
    monkeypatch.setattr(main, "get_static", lambda: static)
    monkeypatch.setattr(main, "load_or_run_forecast", lambda **kw: bundle)
    return static


def _analyze(lat=-9.355, lon=-77.62):
    req = main.GeometryAnalyzeRequest(
        geometry_type="polygon",
        coordinates=[[lat - 0.06, lon - 0.06], [lat - 0.06, lon + 0.06], [lat + 0.06, lon + 0.06], [lat + 0.06, lon - 0.06]],
        label="Marcará",
    )
    return main.analyze_geometry(req)


def test_no_frost_very_dry_gives_non_thermal_recommendation(monkeypatch):
    """Caso Marcará: sin helada + RH 6 % -> no hay alerta crítica de helada ni recomendación térmica;
    el incendio sale como alerta secundaria con su celda 3 km y valor."""
    _fake_env(monkeypatch, T_c=12.0, RH=6.0, W=11.0)
    d = _analyze()
    assert d["kpis"]["tmin_c"] > 0
    assert d["actionable_status"] == "VENTANA SEGURA"
    assert d["alerta_principal_tipo"] == "helada"
    rec = d["recommendation"].lower()
    assert "sin helada" in rec
    assert not any(k in rec for k in ("protección térmica", "protege tu chacra", "riego ligero"))
    tipos = {a["tipo"]: a for a in d["alertas_secundarias"]}
    assert "incendio_sequedad" in tipos and "viento" in tipos
    fire = tipos["incendio_sequedad"]
    assert fire["nivel"] == "alta"
    assert "6%" in fire["texto"] and "celda 3 km" in fire["texto"]
    assert "quema" in fire["recomendacion"].lower()
    assert not any(k in fire["recomendacion"].lower() for k in ("riego", "cobertura", "térmica"))
    assert "celda 3 km" in tipos["viento"]["texto"]


def test_frost_drives_main_alert_and_cites_cell(monkeypatch):
    _fake_env(monkeypatch, T_c=-1.5, RH=60.0, W=2.0)
    d = _analyze()
    assert d["actionable_status"] == "ALERTA CRÍTICA"
    assert "protege tu chacra" in d["recommendation"].lower()
    assert "celda 3 km" in d["recommendation"] and "-2.0 °C" in d["recommendation"]
    assert d["alertas_secundarias"] == []


def test_briefing_no_frost_dry_has_no_thermal_action():
    req = main.BriefingRequest(
        label="Marcará", geometry_type="polygon", elevacion_min_m=2700, elevacion_max_m=3300,
        tmin_c=8.0, tmax_c=24.0, hora_tmin_lima="lun 29/07 05:00", viento_max_ms=6.0,
        wind_chill_min_c=6.0, rh_min_pct=6.0, fire_risk_max=80.0, actionable_status="VENTANA SEGURA",
        use_ollama_if_available=False,
    )
    es = main.generate_bilingual_briefing(req)["spanish"].lower()
    assert "sin helada" in es and "quemas" in es
    assert "protege tu chacra" not in es and "protección contra heladas" not in es
