"""
Hackathon IA PUCP 2026 · Ejemplo mínimo de producto, de punta a punta.

    GFS actual (Open-Meteo)  ->  tu modelo de downscaling  ->  algo útil para un usuario

Este ejemplo hace un "alerta de heladas" muy simple para las próximas 24 horas:
  1. Descarga el pronóstico GFS actual con gfs_forecast.py (grilla 17x13).
  2. Lo pasa por `predecir()`, que devuelve mapas de 3 km (70x35) de RH, T, U, V.
  3. Marca las zonas donde la temperatura baja de 0 °C y guarda un mapa en salidas/.

IMPORTANTE: `predecir()` trae un MODELO DE RELLENO (interpolación + corrección por
altura), solo para que el ejemplo corra. NO es un modelo válido para la hackathon.
Reemplázalo por TU modelo de Kaggle, con los mismos pesos de tu submission final.

Uso:
    pip install -r requirements.txt
    python app/ejemplo_producto.py
    python app/ejemplo_producto.py --desde 2026-07-01 --hasta 2026-07-02   # fechas pasadas

Por defecto lee data/static.npz (se baja de la competencia en Kaggle, pestaña Data)
y guarda el mapa en app/salidas/.
"""
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS, get_gfs_forecast, make_windows

VARS = ["RH", "T", "U", "V"]
APP = Path(__file__).resolve().parent          # carpeta app/
RAIZ = APP.parent                                # raíz del repositorio


# ---------------------------------------------------------------------------
# 1. TU MODELO VA AQUÍ
# ---------------------------------------------------------------------------
def predecir(Xw, static, times=None):
    """
    Entrada:
      Xw     : (N, 3, 17, 13, 4) ventanas [t-2, t-1, t] del GFS, en unidades físicas
               (RH %, T K, U m/s, V m/s). Fila 0 = norte.
      static : contenido de static.npz (dem, wrf_lats, wrf_lons, ...)
      times  : (N,) array de strings 'YYYYMMDDHH' en UTC
    Salida:
      Y      : (N, 70, 35, 4) mapas a 3 km, mismas unidades. FILA 0 = SUR (como en Kaggle).
    """
    import sys
    import torch

    pesos_path = RAIZ / "models" / "modelo.pt"
    stats_path = RAIZ / "models" / "norm_stats.npz"

    if pesos_path.exists() and stats_path.exists():
        sys.path.insert(0, str(RAIZ / "kaggle_pipeline" / "src"))
        from train import (
            StarterCNN,
            TopoResUNet,
            build_sampling_grid,
            build_topographic_features,
            extract_hour_utc,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stats = np.load(stats_path)
        mu_x = stats["mu_x"]
        sd_x = stats["sd_x"]
        clim_trn = stats["clim_trn"]
        anom_sd = stats["anom_sd"]
        arch = str(stats["arch"])
        width = int(stats["width"])

        if arch == "ensemble_v8":
            # Modelo final de Kaggle: inferencia compartida con el backend (mismo preprocesamiento
            # que el entrenamiento, lags ±24 h). Ver app/inti_inference.py.
            from inti_inference import predict_sequence

            X_seq = Xw[:, -1]
            t_seq = times if times is not None else np.array([f"20260101{i%24:02d}" for i in range(len(X_seq))])
            return predict_sequence(X_seq, t_seq, static)

        grid_t = build_sampling_grid(static, device)
        dem = static["dem"].astype(np.float32)

        if arch == "starter_cnn":
            dem_norm_t = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32, device=device)[None, None]
            model = StarterCNN(grid=grid_t, dem_norm=dem_norm_t, w=64).to(device)
        elif arch == "ensemble_v8":
            from train import smooth_2d
            from train_v8 import EnsembleV8

            topo_np = build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"])
            topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
            dem_smooth = smooth_2d(dem, k=9)
            lapse_phys = (-0.0065 * (dem - dem_smooth)).astype(np.float32)
            dem_delta_km_t = torch.tensor(((dem - dem_smooth) / 1000.0).astype(np.float32), dtype=torch.float32, device=device)
            lapse_delta_t = torch.tensor(lapse_phys / float(sd_x[1]), dtype=torch.float32, device=device)

            ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)
            bw = stats["blend_weights"]
            model = EnsembleV8(
                grid=grid_t,
                topo_features=topo_t,
                lapse_delta_norm=lapse_delta_t,
                dem_delta_km=dem_delta_km_t,
                mu_x=torch.tensor(mu_x, dtype=torch.float32, device=device),
                sd_x=torch.tensor(sd_x, dtype=torch.float32, device=device),
                anom_sd=ASD_T,
                blend_weights=bw,
            ).to(device)
        elif arch == "ensemble_v7":
            from train import smooth_2d
            from train_v7 import EnsembleV7, build_v7_temporal_and_spatial_features

            topo_np = build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"])
            topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
            dem_smooth = smooth_2d(dem, k=9)
            lapse_phys = (-0.0065 * (dem - dem_smooth)).astype(np.float32)
            lapse_phys_t = torch.tensor(lapse_phys, dtype=torch.float32, device=device)
            lapse_delta_t = torch.tensor(lapse_phys / float(sd_x[1]), dtype=torch.float32, device=device)

            CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
            ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)
            bw = stats["blend_weights"]
            model = EnsembleV7(
                grid=grid_t,
                topo_features=topo_t,
                lapse_delta_norm=lapse_delta_t,
                lapse_phys_t=lapse_phys_t,
                mu_x=torch.tensor(mu_x, dtype=torch.float32, device=device),
                sd_x=torch.tensor(sd_x, dtype=torch.float32, device=device),
                clim_trn=CLIM_T,
                anom_sd=ASD_T,
                blend_weights=bw,
            ).to(device)
        elif arch in ("physics_unet_v3", "ensemble_3fam_v6"):
            from train_v3 import (
                PhysicsUNet,
                CENTERED_LAGS,
                build_centered_windows_and_daily_mean,
                compute_dewpoint_c,
                smooth_2d,
            )

            topo_np = build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"])
            topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
            dem_smooth = smooth_2d(dem, k=9)
            lapse_delta = (-0.0065 * (dem - dem_smooth)) / float(sd_x[1])
            lapse_delta_t = torch.tensor(lapse_delta, dtype=torch.float32, device=device)

            if arch == "ensemble_3fam_v6":
                from build_ensemble import EnsembleDownscaler

                bw = stats["blend_weights"]
                model = EnsembleDownscaler(
                    grid=grid_t,
                    topo_features=topo_t,
                    lapse_delta_norm=lapse_delta_t,
                    unet_w=64,
                    convnext_w=80,
                    fno_w=48,
                    blend_weights=bw,
                ).to(device)
            else:
                model = PhysicsUNet(
                    grid=grid_t,
                    topo_features=topo_t,
                    lapse_delta_norm=lapse_delta_t,
                    w=width,
                    dropout=0.0,
                ).to(device)
        else:
            topo_np = build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"])
            topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
            model = TopoResUNet(grid=grid_t, topo_features=topo_t, w=width).to(device)

        model.load_state_dict(torch.load(pesos_path, map_location=device, weights_only=True))
        model.eval()

        if times is not None:
            h_utc = extract_hour_utc(times)
        else:
            h_utc = np.zeros(len(Xw), dtype=np.int64)

        CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
        ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)

        with torch.no_grad():
            hb = torch.tensor(h_utc, dtype=torch.long, device=device)
            if arch in ("ensemble_v7", "ensemble_v8"):
                from train_v7 import build_v7_temporal_and_spatial_features

                X_seq = Xw[:, -1]
                t_seq = times if times is not None else np.array([f"20260101{i%24:02d}" for i in range(len(X_seq))])
                W7, M24, S24, GRAD, TD = build_v7_temporal_and_spatial_features(X_seq, t_seq)
                mu_td = float(stats["mu_td"])
                sd_td = float(stats["sd_td"])
                mu_grad = stats["mu_grad"]
                sd_grad = stats["sd_grad"]
                xb = torch.tensor(((W7 - mu_x) / sd_x).astype(np.float32), dtype=torch.float32, device=device)
                mb = torch.tensor(((M24 - mu_x) / sd_x).astype(np.float32), dtype=torch.float32, device=device)
                sb = torch.tensor((S24 / sd_x).astype(np.float32), dtype=torch.float32, device=device)
                gb = torch.tensor(((GRAD - mu_grad) / sd_grad).astype(np.float32), dtype=torch.float32, device=device)
                tb = torch.tensor(((TD - mu_td) / sd_td).astype(np.float32), dtype=torch.float32, device=device)
                Y = model(xb, mb, sb, gb, tb, hb).cpu().numpy()
                Y[..., 0] = np.clip(Y[..., 0], 0.0, 100.0)
                return Y.astype(np.float32)
            if arch in ("physics_unet_v3", "ensemble_3fam_v6"):
                # Reconstruir secuencia horaria X desde el último paso de cada ventana Xw[:, -1]
                X_seq = Xw[:, -1]
                t_seq = times if times is not None else np.array([f"20260101{i%24:02d}" for i in range(len(X_seq))])
                W_c, M24 = build_centered_windows_and_daily_mean(X_seq, t_seq, CENTERED_LAGS)
                Td = compute_dewpoint_c(X_seq[..., 1], X_seq[..., 0])[:, None, :, :]
                mu_td = float(stats["mu_td"])
                sd_td = float(stats["sd_td"])
                xb = torch.tensor(((W_c - mu_x) / sd_x).astype(np.float32), dtype=torch.float32, device=device)
                mb = torch.tensor(((M24 - mu_x) / sd_x).astype(np.float32), dtype=torch.float32, device=device)
                tb = torch.tensor(((Td - mu_td) / sd_td).astype(np.float32), dtype=torch.float32, device=device)
                pred_anom = model(xb, mb, tb, hb.float())
            else:
                Wn = ((Xw - mu_x) / sd_x).astype(np.float32)
                xb = torch.tensor(Wn, dtype=torch.float32, device=device)
                pred_anom = model(xb, hb.float())
            Y = (CLIM_T[hb] + pred_anom * ASD_T).cpu().numpy()
            Y[..., 0] = np.clip(Y[..., 0], 0.0, 100.0)
        return Y.astype(np.float32)

    # Fallback si aún no se ha corrido train.py
    Xt = Xw[:, -1]
    Y = _interpolar_a_wrf(Xt, static["wrf_lats"], static["wrf_lons"])
    dem = static["dem"].astype(np.float32)
    dem_suave = _suavizar(dem, k=9)
    Y[..., 1] -= 0.0065 * (dem - dem_suave)
    return Y


# ---------------------------------------------------------------------------
# 2. LO QUE EL PRODUCTO HACE CON LA SALIDA DEL MODELO
# ---------------------------------------------------------------------------
def alerta_heladas(Y, times, umbral_c=0.0):
    """Para cada píxel: ¿habrá helada en el periodo? y la hora (Lima) de la mínima."""
    T_c = Y[..., 1] - 273.15                               # (N, 70, 35)
    tmin = T_c.min(axis=0)
    hora_min = np.asarray(times)[T_c.argmin(axis=0)]
    helada = tmin < umbral_c
    return tmin, hora_min, helada


def a_hora_lima(t_utc):
    return (datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)).strftime("%a %d/%m %H:00")


def guardar_mapa(tmin, helada, static, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats, lons = static["wrf_lats"], static["wrf_lons"]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    fig, ax = plt.subplots(1, 2, figsize=(9, 8), constrained_layout=True)
    im = ax[0].imshow(tmin, origin="lower", extent=ext, cmap="coolwarm")   # origin=lower: fila 0 = sur
    ax[0].set_title("Temperatura mínima (°C)")
    fig.colorbar(im, ax=ax[0], shrink=0.7)
    ax[1].imshow(static["dem"], origin="lower", extent=ext, cmap="gray")
    ax[1].contourf(np.linspace(ext[0], ext[1], tmin.shape[1]), np.linspace(ext[2], ext[3], tmin.shape[0]),
                   helada.astype(float), levels=[0.5, 1.5], colors=["#4aa3ff"], alpha=0.6)
    ax[1].set_title("Zonas con helada (azul) sobre el relieve")
    for a in ax:
        a.set_xlabel("Longitud"); a.set_ylabel("Latitud")
    fig.suptitle(titulo)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _interpolar_a_wrf(Xt, wrf_lats, wrf_lons):
    """Bilineal por coordenadas: (N, 17, 13, 4) GFS -> (N, 70, 35, 4) WRF (fila 0 = sur)."""
    lat_g = GFS_LATS[::-1]                     # pasar a sur -> norte (creciente)
    Xs = Xt[:, ::-1]
    fi = np.interp(wrf_lats, lat_g, np.arange(len(lat_g)))
    fj = np.interp(wrf_lons, GFS_LONS, np.arange(len(GFS_LONS)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat_g) - 2); di = (fi - i0)[None, :, None, None]
    j0 = np.clip(np.floor(fj).astype(int), 0, len(GFS_LONS) - 2); dj = (fj - j0)[None, None, :, None]
    A = Xs[:, i0][:, :, j0]; B = Xs[:, i0][:, :, j0 + 1]
    C = Xs[:, i0 + 1][:, :, j0]; D = Xs[:, i0 + 1][:, :, j0 + 1]
    return ((A * (1 - dj) + B * dj) * (1 - di) + (C * (1 - dj) + D * dj) * di).astype(np.float32)


def _suavizar(a, k=9):
    p = k // 2
    ap = np.pad(a, p, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            out += ap[di:di + a.shape[0], dj:dj + a.shape[1]]
    return out / (k * k)


def main():
    ap = argparse.ArgumentParser(description="Ejemplo de producto: alerta de heladas")
    ap.add_argument("--static", default=str(RAIZ / "data" / "static.npz"), help="ruta a static.npz (de Kaggle)")
    ap.add_argument("--horas", type=int, default=24, help="horas de pronóstico hacia adelante")
    ap.add_argument("--desde", help="fecha pasada YYYY-MM-DD (opcional)")
    ap.add_argument("--hasta", help="fecha pasada YYYY-MM-DD (opcional)")
    a = ap.parse_args()

    if not os.path.exists(a.static):
        raise SystemExit(f"No encuentro {a.static}. Baja la data de Kaggle a la carpeta data/ (ver README).")
    static = dict(np.load(a.static))
    if a.desde:
        X, times = get_gfs_forecast(start_date=a.desde, end_date=a.hasta or a.desde)
    else:
        # past_hours=24 y +24 h extra: el modelo usa lags t-24 .. t+24 (V7_LAGS)
        X, times = get_gfs_forecast(forecast_hours=a.horas + 24, past_hours=24,
                                    cache_path=str(APP / "gfs_cache.npz"))
    Xw, times = make_windows(X, times)
    Y = predecir(Xw, static, times=times)
    if not a.desde:   # mostrar sólo las horas futuras pedidas (las extra sirven de contexto a los lags)
        sel = slice(24, 24 + a.horas)
        Y, times = Y[sel], times[sel]
    assert Y.shape[1:] == (70, 35, 4), f"predecir() debe devolver (N, 70, 35, 4), devolvió {Y.shape}"

    tmin, hora_min, helada = alerta_heladas(Y, times)
    os.makedirs(APP / "salidas", exist_ok=True)
    ruta = str(APP / "salidas" / "alerta_heladas.png")
    periodo = f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])} (hora de Lima)"
    guardar_mapa(tmin, helada, static, f"Alerta de heladas · {periodo}", ruta)

    print(f"Periodo: {periodo}")
    print(f"Temperatura mínima en el dominio: {tmin.min():.1f} °C")
    print(f"Zonas con helada: {100 * helada.mean():.0f}% del territorio")
    if helada.any():
        i, j = np.unravel_index(tmin.argmin(), tmin.shape)
        print(f"Punto más frío: lat {static['wrf_lats'][i]:.2f}, lon {static['wrf_lons'][j]:.2f}, "
              f"{static['dem'][i, j]:.0f} m, {tmin[i, j]:.1f} °C el {a_hora_lima(hora_min[i, j])}")
    print("Mapa guardado en", ruta)


if __name__ == "__main__":
    main()
