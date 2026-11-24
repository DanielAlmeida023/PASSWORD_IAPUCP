"""
Pipeline Oficial de Entrenamiento y Validación Temporal — Hackathon IA PUCP 2026 (AIRE-ANDINO).

Implementa:
1. Validación temporal estricta (val = ttr >= '2024061600', últimas 2 semanas de junio) con la métrica
   oficial Skill Score sobre la climatología por hora UTC.
2. Baseline Oficial (Net del starter_notebook, ~0.40).
3. Modelo v2 (TopoResUNet + Channel/Spatial SE-Attention + Features Topográficos Físicos + Pérdida directa
   ponderada por el RMSE de climatología para optimizar las 4 variables RH, T, U, V por igual).
4. Registro automático en kaggle_pipeline/experiments/experiments_log.csv, guardado de pesos limpios en
   models/modelo.pt + metadatos de inferencia para el Producto en vivo, y generación de submission.csv.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from seed import seed_everything, DEFAULT_SEED
from experiment_logger import ExperimentLogger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

VARS = ["RH", "T", "U", "V"]
UNITS = ["%", "K", "m/s", "m/s"]


def extract_hour_utc(t: np.ndarray) -> np.ndarray:
    return np.array([int(str(k)[-2:]) for k in t], dtype=np.int64)


def extract_day_of_year(t: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(t.astype(str), format="%Y%m%d%H")
    return ts.dayofyear.values.astype(np.float32)


def compute_climatology(Y: np.ndarray, t: np.ndarray) -> np.ndarray:
    h = extract_hour_utc(t)
    return np.stack([Y[h == k].mean(0) for k in range(24)]).astype(np.float32)  # (24, 70, 35, 4)


def build_time_windows(X: np.ndarray, t: np.ndarray, lags=(2, 1, 0)) -> np.ndarray:
    """
    Construye ventanas temporales [t-2, t-1, t] (o más lags) respetando posibles huecos en `times`.
    Devuelve: (N, len(lags), 17, 13, 4)
    """
    ts = pd.to_datetime(t.astype(str), format="%Y%m%d%H")
    pos = {k: i for i, k in enumerate(ts)}
    out = np.empty((len(X), len(lags)) + X.shape[1:], dtype=np.float32)
    for i, k in enumerate(ts):
        for j, lag in enumerate(lags):
            idx = pos.get(k - pd.Timedelta(hours=lag), None)
            out[i, j] = X[idx] if idx is not None else X[i]
    return out


def smooth_2d(a: np.ndarray, k: int = 9) -> np.ndarray:
    p = k // 2
    ap = np.pad(a, p, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            out += ap[di : di + a.shape[0], dj : dj + a.shape[1]]
    return out / (k * k)


def build_topographic_features(dem: np.ndarray, wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> np.ndarray:
    """
    Construye 10 canales estáticos físicos a partir del DEM (70, 35) y coordenadas WRF (fila 0 = sur):
      0. DEM normalizado
      1. Gradiente Norte-Sur (dy) normalizado -> encajonamiento y exposición meridional del viento V
      2. Gradiente Este-Oeste (dx) normalizado -> exposición zonal del viento U (Cordillera Blanca)
      3. Magnitud de la pendiente (slope) -> zonas de cambio brusco de montaña
      4. Sin(aspect) -> orientación de la ladera Este-Oeste
      5. Cos(aspect) -> orientación de la ladera Norte-Sur
      6. Anomalía local de altura (radio ~9 km, k=3): valle vs pico inmediato (TPI corto)
      7. Anomalía mesoescalar de altura (radio ~27 km, k=9): lo que el GFS no resuelve (TPI GFS)
      8. Latitud normalizada [-1, 1]
      9. Longitud normalizada [-1, 1]
    Devuelve tensor numpy: (10, 70, 35)
    """
    dem_f = dem.astype(np.float32)
    dem_norm = (dem_f - dem_f.mean()) / (dem_f.std() + 1e-6)

    # Gradientes espaciales (aprox 3 km por celda)
    dy, dx = np.gradient(dem_f)
    dx_norm = (dx - dx.mean()) / (dx.std() + 1e-6)
    dy_norm = (dy - dy.mean()) / (dy.std() + 1e-6)

    slope = np.sqrt(dx**2 + dy**2)
    slope_norm = (slope - slope.mean()) / (slope.std() + 1e-6)

    aspect = np.arctan2(dy, dx)
    sin_aspect = np.sin(aspect).astype(np.float32)
    cos_aspect = np.cos(aspect).astype(np.float32)

    tpi_local = dem_f - smooth_2d(dem_f, k=3)
    tpi_local_norm = (tpi_local - tpi_local.mean()) / (tpi_local.std() + 1e-6)

    tpi_meso = dem_f - smooth_2d(dem_f, k=9)
    tpi_meso_norm = (tpi_meso - tpi_meso.mean()) / (tpi_meso.std() + 1e-6)

    LAT, LON = np.meshgrid(wrf_lats, wrf_lons, indexing="ij")
    lat_norm = ((LAT - LAT.mean()) / (LAT.std() + 1e-6)).astype(np.float32)
    lon_norm = ((LON - LON.mean()) / (LON.std() + 1e-6)).astype(np.float32)

    topo = np.stack(
        [
            dem_norm,
            dy_norm,
            dx_norm,
            slope_norm,
            sin_aspect,
            cos_aspect,
            tpi_local_norm,
            tpi_meso_norm,
            lat_norm,
            lon_norm,
        ],
        axis=0,
    ).astype(np.float32)
    return topo  # (10, 70, 35)


def build_sampling_grid(st: Dict[str, np.ndarray], device: str) -> torch.Tensor:
    glat, glon = st["gfs_lats"], st["gfs_lons"]  # GFS: fila 0 = norte
    wlat, wlon = st["wrf_lats"], st["wrf_lons"]  # WRF: fila 0 = sur
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    grid = np.stack([gx, gy], axis=-1)[None].astype(np.float32)  # (1, 70, 35, 2)
    return torch.tensor(grid, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Arquitecturas de Modelos
# ---------------------------------------------------------------------------
class StarterCNN(nn.Module):
    """Baseline oficial del starter_notebook.ipynb (score ~0.40)."""

    def __init__(self, grid: torch.Tensor, dem_norm: torch.Tensor, w: int = 64):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("dem", dem_norm)
        self.enc = nn.Sequential(
            nn.Conv2d(14, w, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1),
            nn.GELU(),
        )
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(
            nn.Conv2d(w + 9, w, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, 4, 1),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 17, 13, 4), h: (B,)
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        ang = 2 * np.pi * h / 24.0
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
        f = self.enc(torch.cat([x, hh], 1))
        f = F.grid_sample(f, self.grid.expand(B, -1, -1, -1), align_corners=True)
        f = torch.cat([f, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
        return self.dec(f).permute(0, 2, 3, 1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention para ponderar canales dinámicamente."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Bloque Residual con GroupNorm, GELU y atención Squeeze-and-Excitation."""

    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.block(x)))


class TopoResUNet(nn.Module):
    """
    Arquitectura v2 para Downscaling Climático en la Cordillera Blanca:
    1. Extrae características físicas del GFS en 17x13 (incluyendo velocidad escalar del viento,
       diferencias temporales dX = X_t - X_{t-2} y tendencias horarias).
    2. Codifica en baja resolución con bloques residuales + interpolación física diferenciable (grid_sample)
       tanto de las features profundas como del GFS crudo de la hora t (skip connection físico).
    3. Combina en 70x35 con los 10 canales topográficos del DEM + mapa estático aprendible por celda.
    4. Pasa por una estructura U-Net multi-escala (70x35 -> 35x18 -> 70x35) y cabezales especializados
       para (RH, T) y para el vector de viento (U, V) modulado por pendiente y orientación.
    """

    def __init__(self, grid: torch.Tensor, topo_features: torch.Tensor, w: int = 96):
        super().__init__()
        self.register_buffer("grid", grid)  # (1, 70, 35, 2)
        self.register_buffer("topo", topo_features[None])  # (1, 10, 70, 35)

        # Canales de entrada GFS en 17x13:
        # 3 pasos * 4 vars = 12
        # + 3 pasos * 1 wind_speed (sqrt(U^2 + V^2)) = 3
        # + delta temporal (X_t - X_{t-2}) = 4
        # + delta 1h (X_t - X_{t-1}) = 4
        # + hora UTC (sin, cos) = 2
        # Total = 25 canales en 17x13
        in_gfs_ch = 25
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(in_gfs_ch, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            ResBlock(w, dropout=0.05),
            ResBlock(w, dropout=0.05),
        )

        # Embedding estático aprendido en 70x35 (captura microclimas locales de cada píxel)
        self.learned_static = nn.Parameter(torch.zeros(1, 16, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.02)

        # En 70x35 fusionamos:
        # w (features GFS interpoladas) + 12 (GFS crudo interpolado bilinealmente) + 10 (topo físico) + 16 (learned_static) + 2 (hora)
        fused_ch = w + 12 + 10 + 16 + 2

        self.highres_stem = nn.Sequential(
            nn.Conv2d(fused_ch, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            ResBlock(w, dropout=0.05),
        )

        # Rama multi-escala U-Net (captura circulación de valle a escala regional 35x17)
        self.down = nn.Sequential(
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, w * 2),
            nn.GELU(),
            ResBlock(w * 2, dropout=0.08),
            ResBlock(w * 2, dropout=0.08),
        )

        self.up_fuse = nn.Sequential(
            nn.Conv2d(w + w * 2, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            ResBlock(w, dropout=0.05),
            ResBlock(w, dropout=0.05),
        )

        # Cabezales especializados: Termodinámico (RH, T) y Dinámico de Viento (U, V)
        self.head_thermo = nn.Sequential(
            nn.Conv2d(w + 10, w // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, 1),  # RH, T
        )
        self.head_wind = nn.Sequential(
            nn.Conv2d(w + 10, w // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, 1),  # U, V
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 17, 13, 4) normalizado [t-2, t-1, t], h: (B,) hora UTC
        B = x.shape[0]
        # Variables derivadas en GFS
        u_gfs, v_gfs = x[..., 2], x[..., 3]
        wspd = torch.sqrt(u_gfs**2 + v_gfs**2 + 1e-6).unsqueeze(-1)  # (B, 3, 17, 13, 1)

        dx_2h = x[:, 2] - x[:, 0]  # (B, 17, 13, 4)
        dx_1h = x[:, 2] - x[:, 1]  # (B, 17, 13, 4)

        x_flat = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        wspd_flat = wspd.permute(0, 1, 4, 2, 3).reshape(B, 3, 17, 13)
        dx2_flat = dx_2h.permute(0, 3, 1, 2)
        dx1_flat = dx_1h.permute(0, 3, 1, 2)

        ang = 2.0 * np.pi * h / 24.0
        h_vec = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)  # (B, 2)
        hh_low = h_vec[..., None, None].expand(B, 2, 17, 13)

        gfs_in = torch.cat([x_flat, wspd_flat, dx2_flat, dx1_flat, hh_low], dim=1)  # (B, 25, 17, 13)
        f_low = self.gfs_stem(gfs_in)  # (B, w, 17, 13)

        # Interpolación a la grilla WRF 70x35 (invierte norte/sur automáticamente vía self.grid)
        grid_b = self.grid.expand(B, -1, -1, -1)
        f_up = F.grid_sample(f_low, grid_b, mode="bilinear", align_corners=True)  # (B, w, 70, 35)
        raw_up = F.grid_sample(x_flat, grid_b, mode="bilinear", align_corners=True)  # (B, 12, 70, 35)

        topo_b = self.topo.expand(B, -1, -1, -1)  # (B, 10, 70, 35)
        static_b = self.learned_static.expand(B, -1, -1, -1)  # (B, 16, 70, 35)
        hh_high = h_vec[..., None, None].expand(B, 2, 70, 35)

        fused = torch.cat([f_up, raw_up, topo_b, static_b, hh_high], dim=1)
        s1 = self.highres_stem(fused)  # (B, w, 70, 35)

        s2 = self.down(s1)  # (B, 2w, 35, 18)
        s2_up = F.interpolate(s2, size=(70, 35), mode="bilinear", align_corners=True)

        dec = self.up_fuse(torch.cat([s1, s2_up], dim=1))  # (B, w, 70, 35)
        dec_topo = torch.cat([dec, topo_b], dim=1)

        out_thermo = self.head_thermo(dec_topo)  # (B, 2, 70, 35)
        out_wind = self.head_wind(dec_topo)  # (B, 2, 70, 35)

        out = torch.cat([out_thermo, out_wind], dim=1)  # (B, 4, 70, 35)
        return out.permute(0, 2, 3, 1)  # (B, 70, 35, 4)


# ---------------------------------------------------------------------------
# Evaluación Oficial y Entrenamiento
# ---------------------------------------------------------------------------
def evaluate_skill_score(
    pred: np.ndarray, true: np.ndarray, clim_rmse_val: np.ndarray
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=(0, 1, 2)))
    rel_skill = 1.0 - (rmse / clim_rmse_val)
    total_skill = float(rel_skill.mean())
    rmse_dict = {k: round(float(v), 4) for k, v in zip(VARS, rmse)}
    skill_dict = {k: round(float(v), 4) for k, v in zip(VARS, rel_skill)}
    return total_skill, rmse_dict, skill_dict


def train_and_evaluate(
    arch: str = "toporesunet",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    width: int = 96,
    seed: int = DEFAULT_SEED,
    save_submission: bool = True,
) -> float:
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Dispositivo: {device.upper()} | Arquitectura: {arch} | Seed: {seed} | Épocas: {epochs}")

    tr = np.load(DATA_DIR / "train.npz")
    te = np.load(DATA_DIR / "test.npz")
    st = dict(np.load(DATA_DIR / "static.npz"))

    Xtr, Ytr, ttr = tr["X"], tr["Y"], tr["times"].astype(str)
    Xte, tte = te["X"], te["times"].astype(str)
    dem = st["dem"]

    # Corte temporal estricto (últimas 2 semanas de junio como validación)
    val_mask = ttr >= "2024061600"
    trn_mask = ~val_mask

    clim_trn = compute_climatology(Ytr[trn_mask], ttr[trn_mask])
    h_tr = extract_hour_utc(ttr)
    h_te = extract_hour_utc(tte)

    clim_rmse_val = np.sqrt(((clim_trn[h_tr[val_mask]] - Ytr[val_mask]) ** 2).mean(axis=(0, 1, 2)))
    base_clim_score, base_rmse, _ = evaluate_skill_score(clim_trn[h_tr[val_mask]], Ytr[val_mask], clim_rmse_val)
    print(f"[Validación Temporal] Horas train: {trn_mask.sum()} | Horas val: {val_mask.sum()}")
    print(f"[Baseline Climatología] Skill Score: {base_clim_score:.4f} | RMSE Climatología: {base_rmse}")

    # Ventanas de 3 horas [t-2, t-1, t]
    Wtr = build_time_windows(Xtr, ttr, lags=(2, 1, 0))
    Wte = build_time_windows(Xte, tte, lags=(2, 1, 0))

    # Estadísticas de normalización (calculadas SOLO sobre train para evitar data leakage)
    mu_x = Xtr[trn_mask].mean(axis=(0, 1, 2)).astype(np.float32)
    sd_x = Xtr[trn_mask].std(axis=(0, 1, 2)).astype(np.float32)
    anom_sd = (Ytr[trn_mask] - clim_trn[h_tr[trn_mask]]).std(axis=(0, 1, 2)).astype(np.float32)

    grid_t = build_sampling_grid(st, device)
    topo_np = build_topographic_features(dem, st["wrf_lats"], st["wrf_lons"])
    topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
    dem_norm_t = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32, device=device)[None, None]

    if arch == "starter_cnn":
        model = StarterCNN(grid=grid_t, dem_norm=dem_norm_t, w=64).to(device)
    elif arch == "toporesunet":
        model = TopoResUNet(grid=grid_t, topo_features=topo_t, w=width).to(device)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] Parámetros entrenables: {n_params:,}")

    CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
    ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)
    CLIM_RMSE_T = torch.tensor(clim_rmse_val, dtype=torch.float32, device=device)

    def prep_batch(W_sub: np.ndarray, h_sub: np.ndarray):
        Wn = (W_sub - mu_x) / sd_x
        return (
            torch.tensor(Wn, dtype=torch.float32, device=device),
            torch.tensor(h_sub, dtype=torch.long, device=device),
        )

    def predict_array(net: nn.Module, W_all: np.ndarray, h_all: np.ndarray, bs: int = 128) -> np.ndarray:
        net.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(W_all), bs):
                xb, hb = prep_batch(W_all[s : s + bs], h_all[s : s + bs])
                pred_anom = net(xb, hb.float())
                y_hat = CLIM_T[hb] + pred_anom * ASD_T
                y_np = y_hat.cpu().numpy()
                # Restricción física: Humedad Relativa RH (%) está acotada entre [0, 100]
                y_np[..., 0] = np.clip(y_np[..., 0], 0.0, 100.0)
                preds.append(y_np)
        return np.concatenate(preds, axis=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    itr = np.where(trn_mask)[0]
    steps_per_epoch = int(np.ceil(len(itr) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15
    )

    # Pesos de pérdida alineados directamente con el Skill Score oficial:
    # Como pred_anom está escalado por ASD_T, el error físico es pred_anom * ASD_T.
    # Dividir entre CLIM_RMSE_T alinea el MSE exactamente con (RMSE_v / RMSE_clim_v)^2.
    var_loss_weights = (ASD_T / CLIM_RMSE_T) ** 2
    var_loss_weights = var_loss_weights / var_loss_weights.mean()

    best_score = -1e9
    best_state = None
    best_rmse_dict = {}
    best_skill_dict = {}

    for ep in range(epochs):
        model.train()
        np.random.shuffle(itr)
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for s in range(0, len(itr), batch_size):
            b_idx = itr[s : s + batch_size]
            xb, hb = prep_batch(Wtr[b_idx], h_tr[b_idx])
            yb_true = torch.tensor(Ytr[b_idx], dtype=torch.float32, device=device)
            yb_anom = (yb_true - CLIM_T[hb]) / ASD_T

            pred_anom = model(xb, hb.float())
            if arch == "starter_cnn":
                loss = F.mse_loss(pred_anom, yb_anom)
            else:
                # Combinación de MSE ponderado por métrica oficial + SmoothL1 para robustez en extremos
                sq_err = ((pred_anom - yb_anom) ** 2) * var_loss_weights[None, None, None, :]
                huber_err = F.smooth_l1_loss(pred_anom, yb_anom, reduction="none", beta=0.5) * var_loss_weights[None, None, None, :]
                loss = 0.75 * sq_err.mean() + 0.25 * huber_err.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

        val_preds = predict_array(model, Wtr[val_mask], h_tr[val_mask])
        sc, rmse_d, skill_d = evaluate_skill_score(val_preds, Ytr[val_mask], clim_rmse_val)
        dt = time.time() - t0

        marker = ""
        if sc > best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_rmse_dict = rmse_d
            best_skill_dict = skill_d
            marker = " ★ BEST"

        print(
            f"Época {ep+1:02d}/{epochs:02d} · loss {running_loss/n_batches:.4f} · "
            f"Skill Val: {sc:.4f} · Skill/var: {skill_d} · RMSE: {rmse_d} · {dt:.1f}s{marker}"
        )

    print("=" * 75)
    print(f"🏆 MEJOR SKILL SCORE EN VALIDACIÓN ({arch}): {best_score:.4f}")
    print(f"   Skill por variable : {best_skill_dict}")
    print(f"   RMSE por variable  : {best_rmse_dict}")
    print("=" * 75)

    model.load_state_dict(best_state)
    val_oof = predict_array(model, Wtr[val_mask], h_tr[val_mask])

    # Re-calcular climatología sobre TODO train (feb-jun) para inferencia final o conservar estadísticas
    test_preds = predict_array(model, Wte, h_te)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    exp_id = logger.log_experiment(
        model_name=arch,
        metric_name="skill_score",
        fold_scores=[best_score],
        params={
            "arch": arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "width": width,
            "n_params": n_params,
            "val_rmse_per_var": best_rmse_dict,
            "val_skill_per_var": best_skill_dict,
        },
        features=["GFS_3h_window_RH_T_U_V", "DEM_10ch_Topography", "UTC_Hour_SinCos", "Learned_Spatial_Static"],
        oof_preds=val_oof,
        test_preds=test_preds,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=seed,
        notes=f"Skill={best_score:.4f} | Skills={best_skill_dict}",
    )

    if save_submission:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # 1. Guardar solo state_dict en models/modelo.pt (como exige el template oficial)
        weights_path = MODELS_DIR / "modelo.pt"
        torch.save(best_state, weights_path)
        size_mb = os.path.getsize(weights_path) / 1e6
        print(f"[Export] Pesos limpios guardados en: {weights_path} ({size_mb:.2f} MB)")

        # 2. Guardar estadísticas de normalización y climatología en models/norm_stats.npz para el Producto en vivo
        np.savez_compressed(
            MODELS_DIR / "norm_stats.npz",
            mu_x=mu_x,
            sd_x=sd_x,
            clim_trn=clim_trn,
            anom_sd=anom_sd,
            clim_rmse_val=clim_rmse_val,
            arch=np.array(arch),
            width=np.array(width),
            best_score=np.array(best_score),
            exp_id=np.array(exp_id),
        )

        # 3. Generar submission.csv oficial (639,450 filas evaluadas cada 3 horas)
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
        pos = {k: i for i, k in enumerate(tte)}
        parts = sample["id"].str.split("_", expand=True)
        ti = parts[0].map(pos).values
        r = parts[1].astype(int).values
        c = parts[2].astype(int).values

        sub = pd.DataFrame({"id": sample["id"]})
        for k, v in enumerate(VARS):
            sub[v] = test_preds[ti, r, c, k].round(3)

        sub_path = ROOT_DIR / "submission.csv"
        sub.to_csv(sub_path, index=False)
        exp_sub_path = EXPERIMENTS_DIR / exp_id / "submission.csv"
        sub.to_csv(exp_sub_path, index=False)
        print(f"[Export] submission.csv generado en: {sub_path} (shape={sub.shape})")

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Downscaling Climático — Hackathon IA PUCP 2026")
    parser.add_argument("--arch", type=str, default="toporesunet", choices=["starter_cnn", "toporesunet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train_and_evaluate(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        seed=args.seed,
        save_submission=not args.no_save,
    )
