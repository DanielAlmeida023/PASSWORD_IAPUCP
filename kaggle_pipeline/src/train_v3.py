"""
Experimento v3 — PhysicsUNet para Downscaling Climático (Hackathon IA PUCP 2026 · AIRE-ANDINO).

Mejoras clave sobre TopoResUNet v2 (Val Skill 0.3459):
1. Ventana temporal centrada de GFS [t-6, t-3, t-1, t, t+1, t+3, t+6] + promedio móvil de 24h
   (detecta desfases horarios del GFS y el régimen estacional seco/húmedo automáticamente).
2. Pistas físicas de meteorología andina (100% derivadas de la data oficial):
   - Punto de rocío (fórmula de Magnus) para estabilizar humedad relativa (RH) en Puna/Alta Montaña.
   - Prior de temperatura por gradiente térmico (-6.5 °C/km sobre DEM - DEM_suave).
   - Forzamiento orográfico de viento (V_GFS · grad(DEM)) y componente canalizada en valles.
3. Pérdida directa alineada al Skill Score oficial (calculada SOLO con estadísticas de train, cero leakage)
   + ponderación estacional (menor peso a febrero anómalo, mayor peso a mayo-junio seco).
4. Arquitectura U-Net multi-escala (3 niveles) compacta (~1.0M params vs 2.9M), ruido gaussiano en GFS
   como data augmentation y EMA (Exponential Moving Average) de pesos para frenar el overfitting.
"""

import argparse
import copy
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
from train import (
    VARS,
    extract_hour_utc,
    compute_climatology,
    smooth_2d,
    build_topographic_features,
    build_sampling_grid,
    evaluate_skill_score,
    SEBlock,
)

from paths import ROOT_DIR, DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR  # env PASSWORD_DATA_DIR / PASSWORD_OUT_DIR

CENTERED_LAGS = (-6, -3, -1, 0, 1, 3, 6)


def compute_dewpoint_c(T_k: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    """Fórmula de Magnus para punto de rocío (°C) a partir de T (K) y RH (%)."""
    T_c = T_k - 273.15
    rh = np.clip(RH_pct, 1.0, 100.0) / 100.0
    a, b = 17.625, 243.04
    gamma = np.log(rh) + (a * T_c) / (b + T_c)
    td = (b * gamma) / (a - gamma)
    return td.astype(np.float32)


def build_centered_windows_and_daily_mean(
    X: np.ndarray, t: np.ndarray, lags: Tuple[int, ...] = CENTERED_LAGS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye:
      - W: (N, len(lags), 17, 13, 4) con desplazamientos centrados [t-6, t-3, t-1, t, t+1, t+3, t+6].
      - M24: (N, 17, 13, 4) promedio móvil de las horas disponibles en [t-12h, t+12h] (régimen sinóptico).
    Si falta una hora por hueco o borde, usa la hora disponible más cercana dentro del conjunto.
    """
    ts = pd.to_datetime(t.astype(str), format="%Y%m%d%H")
    ts_ns = ts.values.astype("datetime64[h]").astype(np.int64)
    pos = {int(h): i for i, h in enumerate(ts_ns)}

    N = len(X)
    W = np.empty((N, len(lags)) + X.shape[1:], dtype=np.float32)
    M24 = np.empty_like(X, dtype=np.float32)

    for i, cur_h in enumerate(ts_ns):
        for j, lag in enumerate(lags):
            target_h = int(cur_h + lag)
            idx = pos.get(target_h, None)
            if idx is None:
                # Buscar la hora más cercana dentro de ±3h o caer a X[i]
                for delta in (1, -1, 2, -2, 3, -3):
                    idx = pos.get(target_h + delta, None)
                    if idx is not None:
                        break
            W[i, j] = X[idx] if idx is not None else X[i]

        # Media móvil 24h centrada
        win_idxs = [pos[int(cur_h + d)] for d in range(-12, 13) if int(cur_h + d) in pos]
        M24[i] = X[win_idxs].mean(axis=0) if win_idxs else X[i]

    return W, M24


class CompactResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.10):
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


class PhysicsUNet(nn.Module):
    """
    Arquitectura v3: U-Net Multi-Escala (3 niveles) Compacta y Guiada por Física Andina.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        w: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.register_buffer("grid", grid)  # (1, 70, 35, 2)
        self.register_buffer("topo", topo_features[None])  # (1, 10, 70, 35)
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])  # (1, 1, 70, 35)

        # Entradas en 17x13:
        # 7 lags * 4 vars = 28
        # + M24 (media 24h) = 4
        # + Td(t) (punto de rocío normalizado) = 1
        # + wind_speed en [t-3, t, t+3] = 3
        # + tendencias: X(t)-X(t-3), X(t+3)-X(t), X(t)-M24 = 12
        # + hora UTC (sin, cos) = 2
        # Total = 50 canales
        in_gfs_ch = 50
        self.gfs_enc = nn.Sequential(
            nn.Conv2d(in_gfs_ch, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            CompactResBlock(w, dropout=dropout * 0.5),
            CompactResBlock(w, dropout=dropout * 0.5),
        )

        # Embedding estático aprendido más compacto para evitar overfitting
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        # Fusión en 70x35:
        # w (GFS deep features)
        # + 8 (GFS X(t) y M24 interpolados crudos)
        # + 1 (Td interpolado)
        # + 1 (lapse_delta: T_gfs_interp - 0.0065*(DEM-DEM_suave))
        # + 2 (forzamiento orográfico: V·grad(DEM) y V x grad(DEM))
        # + 10 (topo físico)
        # + 12 (learned_static)
        # + 2 (hora sin, cos)
        fused_ch = w + 8 + 1 + 1 + 2 + 10 + 12 + 2

        # Nivel 1: 70x35
        self.enc1 = nn.Sequential(
            nn.Conv2d(fused_ch, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            CompactResBlock(w, dropout=dropout),
        )
        # Nivel 2: 35x18
        self.enc2 = nn.Sequential(
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, w * 2),
            nn.GELU(),
            CompactResBlock(w * 2, dropout=dropout),
        )
        # Nivel 3 (cuello de botella regional): 18x9
        self.enc3 = nn.Sequential(
            nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, w * 2),
            nn.GELU(),
            CompactResBlock(w * 2, dropout=dropout * 1.25),
        )

        # Decodificador multi-escala con skip connections
        self.dec2 = nn.Sequential(
            nn.Conv2d(w * 2 + w * 2, w * 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, w * 2),
            nn.GELU(),
            CompactResBlock(w * 2, dropout=dropout),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(w + w * 2, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            CompactResBlock(w, dropout=dropout * 0.5),
        )

        # Cabezales especializados: Termodinámico (RH, T) y Dinámico de Viento (U, V)
        head_in = w + 10 + 2
        self.head_thermo = nn.Sequential(
            nn.Conv2d(head_in, w // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, 1),
        )
        self.head_wind = nn.Sequential(
            nn.Conv2d(head_in, w // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, 1),
        )

    def forward(
        self,
        w_norm: torch.Tensor,  # (B, 7, 17, 13, 4) -> lags (-6, -3, -1, 0, 1, 3, 6)
        m24_norm: torch.Tensor,  # (B, 17, 13, 4)
        td_norm: torch.Tensor,  # (B, 1, 17, 13)
        h: torch.Tensor,  # (B,) hora UTC
    ) -> torch.Tensor:
        B = w_norm.shape[0]

        # Índice 3 corresponde a lag=0 (hora t), índice 1 a t-3, índice 5 a t+3
        xt = w_norm[:, 3]  # (B, 17, 13, 4)
        xm3 = w_norm[:, 1]
        xp3 = w_norm[:, 5]

        # Velocidades de viento en t-3, t, t+3
        wspd_list = []
        for idx in (1, 3, 5):
            u_i, v_i = w_norm[:, idx, :, :, 2], w_norm[:, idx, :, :, 3]
            wspd_list.append(torch.sqrt(u_i**2 + v_i**2 + 1e-6))
        wspd = torch.stack(wspd_list, dim=1)  # (B, 3, 17, 13)

        # Tendencias temporales y anomalía diurna
        d_prev = (xt - xm3).permute(0, 3, 1, 2)  # (B, 4, 17, 13)
        d_next = (xp3 - xt).permute(0, 3, 1, 2)
        d_diurnal = (xt - m24_norm).permute(0, 3, 1, 2)

        w_flat = w_norm.permute(0, 1, 4, 2, 3).reshape(B, 28, 17, 13)
        m24_flat = m24_norm.permute(0, 3, 1, 2)  # (B, 4, 17, 13)

        ang = 2.0 * np.pi * h / 24.0
        h_vec = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)  # (B, 2)
        hh_low = h_vec[..., None, None].expand(B, 2, 17, 13)

        gfs_in = torch.cat([w_flat, m24_flat, td_norm, wspd, d_prev, d_next, d_diurnal, hh_low], dim=1)
        f_low = self.gfs_enc(gfs_in)  # (B, w, 17, 13)

        # Interpolación diferenciable GFS -> WRF (70x35)
        grid_b = self.grid.expand(B, -1, -1, -1)
        f_up = F.grid_sample(f_low, grid_b, mode="bilinear", align_corners=True)
        raw_cat = torch.cat([xt.permute(0, 3, 1, 2), m24_flat, td_norm], dim=1)  # (B, 9, 17, 13)
        raw_up = F.grid_sample(raw_cat, grid_b, mode="bilinear", align_corners=True)  # (B, 9, 70, 35)

        topo_b = self.topo.expand(B, -1, -1, -1)  # (B, 10, 70, 35)
        # topo[:, 1] = dy_norm, topo[:, 2] = dx_norm
        u_up = raw_up[:, 2:3]
        v_up = raw_up[:, 3:4]
        dy_n = topo_b[:, 1:2]
        dx_n = topo_b[:, 2:3]

        # Forzamiento orográfico (barlovento/sotavento) y flujo paralelo al valle
        oro_dot = u_up * dx_n + v_up * dy_n
        oro_cross = u_up * dy_n - v_up * dx_n
        oro_feats = torch.cat([oro_dot, oro_cross], dim=1)  # (B, 2, 70, 35)

        # Prior de temperatura por lapse-rate
        t_prior = raw_up[:, 1:2] + self.lapse_delta.expand(B, -1, -1, -1)

        static_b = self.learned_static.expand(B, -1, -1, -1)
        hh_high = h_vec[..., None, None].expand(B, 2, 70, 35)

        fused = torch.cat([f_up, raw_up, t_prior, oro_feats, topo_b, static_b, hh_high], dim=1)

        s1 = self.enc1(fused)  # (B, w, 70, 35)
        s2 = self.enc2(s1)  # (B, 2w, 35, 18)
        s3 = self.enc3(s2)  # (B, 2w, 18, 9)

        u2 = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([s2, u2], dim=1))

        u1 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([s1, u1], dim=1))

        head_in = torch.cat([d1, topo_b, oro_feats], dim=1)
        out_thermo = self.head_thermo(head_in)  # RH, T
        out_wind = self.head_wind(head_in)  # U, V

        out = torch.cat([out_thermo, out_wind], dim=1)  # (B, 4, 70, 35)
        return out.permute(0, 2, 3, 1)  # (B, 70, 35, 4)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.992) -> None:
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_v3(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    width: int = 64,
    dropout: float = 0.12,
    noise_std: float = 0.04,
    seed: int = DEFAULT_SEED,
    prev_best_score: float = 0.3459,
) -> float:
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train v3] Dispositivo: {device.upper()} | Arquitectura: PhysicsUNet | Width: {width} | Seed: {seed}")

    tr = np.load(DATA_DIR / "train.npz")
    te = np.load(DATA_DIR / "test.npz")
    st = dict(np.load(DATA_DIR / "static.npz"))

    Xtr, Ytr, ttr = tr["X"], tr["Y"], tr["times"].astype(str)
    Xte, tte = te["X"], te["times"].astype(str)
    dem = st["dem"].astype(np.float32)

    val_mask = ttr >= "2024061600"
    trn_mask = ~val_mask

    h_tr = extract_hour_utc(ttr)
    h_te = extract_hour_utc(tte)

    clim_trn = compute_climatology(Ytr[trn_mask], ttr[trn_mask])
    clim_rmse_val = np.sqrt(((clim_trn[h_tr[val_mask]] - Ytr[val_mask]) ** 2).mean(axis=(0, 1, 2)))
    # Estadística de RMSE de climatología SOLO sobre train para la función de pérdida (cero leakage de val)
    clim_rmse_trn = np.sqrt(((clim_trn[h_tr[trn_mask]] - Ytr[trn_mask]) ** 2).mean(axis=(0, 1, 2))).astype(np.float32)

    print("[Preproc] Construyendo ventanas centradas [t-6..t+6] y media móvil 24h...")
    Wtr, M24_tr = build_centered_windows_and_daily_mean(Xtr, ttr, CENTERED_LAGS)
    Wte, M24_te = build_centered_windows_and_daily_mean(Xte, tte, CENTERED_LAGS)

    # Punto de rocío (Td) en 17x13 para hora t
    Td_tr = compute_dewpoint_c(Xtr[..., 1], Xtr[..., 0])[:, None, :, :]  # (N, 1, 17, 13)
    Td_te = compute_dewpoint_c(Xte[..., 1], Xte[..., 0])[:, None, :, :]

    # Normalización SOLO con train
    mu_x = Xtr[trn_mask].mean(axis=(0, 1, 2)).astype(np.float32)
    sd_x = Xtr[trn_mask].std(axis=(0, 1, 2)).astype(np.float32)
    mu_td = float(Td_tr[trn_mask].mean())
    sd_td = float(Td_tr[trn_mask].std() + 1e-6)
    anom_sd = (Ytr[trn_mask] - clim_trn[h_tr[trn_mask]]).std(axis=(0, 1, 2)).astype(np.float32)

    # Ponderación estacional por mes (febrero anómalo pesa menos, mayo-junio seco pesa más)
    months_tr = pd.to_datetime(ttr, format="%Y%m%d%H").month.values
    month_w_map = {2: 0.55, 3: 0.80, 4: 0.90, 5: 1.15, 6: 1.40}
    sample_weights = np.array([month_w_map.get(int(m), 1.0) for m in months_tr], dtype=np.float32)
    sample_weights[trn_mask] /= sample_weights[trn_mask].mean()

    # Features topográficos y lapse-rate delta
    grid_t = build_sampling_grid(st, device)
    topo_np = build_topographic_features(dem, st["wrf_lats"], st["wrf_lons"])
    topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)

    dem_smooth = smooth_2d(dem, k=9)
    lapse_delta = (-0.0065 * (dem - dem_smooth)) / float(sd_x[1])  # en unidades de sd_x[T]
    lapse_delta_t = torch.tensor(lapse_delta, dtype=torch.float32, device=device)

    model = PhysicsUNet(
        grid=grid_t,
        topo_features=topo_t,
        lapse_delta_norm=lapse_delta_t,
        w=width,
        dropout=dropout,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] PhysicsUNet parámetros entrenables: {n_params:,}")

    CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
    ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)
    CLIM_RMSE_TRN_T = torch.tensor(clim_rmse_trn, dtype=torch.float32, device=device)

    # Pre-normalizar en CPU para acelerar cada época
    Wtr_n = ((Wtr - mu_x) / sd_x).astype(np.float32)
    M24_tr_n = ((M24_tr - mu_x) / sd_x).astype(np.float32)
    Td_tr_n = ((Td_tr - mu_td) / sd_td).astype(np.float32)

    Wte_n = ((Wte - mu_x) / sd_x).astype(np.float32)
    M24_te_n = ((M24_te - mu_x) / sd_x).astype(np.float32)
    Td_te_n = ((Td_te - mu_td) / sd_td).astype(np.float32)

    def predict_array(net: nn.Module, Wn, Mn, Tn, h_all, bs: int = 128) -> np.ndarray:
        net.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Wn), bs):
                xb = torch.tensor(Wn[s : s + bs], dtype=torch.float32, device=device)
                mb = torch.tensor(Mn[s : s + bs], dtype=torch.float32, device=device)
                tb = torch.tensor(Tn[s : s + bs], dtype=torch.float32, device=device)
                hb = torch.tensor(h_all[s : s + bs], dtype=torch.long, device=device)
                pred_anom = net(xb, mb, tb, hb.float())
                y_hat = CLIM_T[hb] + pred_anom * ASD_T
                y_np = y_hat.cpu().numpy()
                y_np[..., 0] = np.clip(y_np[..., 0], 0.0, 100.0)
                preds.append(y_np)
        return np.concatenate(preds, axis=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    itr = np.where(trn_mask)[0]
    steps_per_epoch = int(np.ceil(len(itr) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15
    )

    best_score = -1e9
    best_state = None
    best_rmse_dict: Dict[str, float] = {}
    best_skill_dict: Dict[str, float] = {}
    best_source = "live"

    for ep in range(epochs):
        model.train()
        np.random.shuffle(itr)
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for s in range(0, len(itr), batch_size):
            b_idx = itr[s : s + batch_size]
            xb = torch.tensor(Wtr_n[b_idx], dtype=torch.float32, device=device)
            mb = torch.tensor(M24_tr_n[b_idx], dtype=torch.float32, device=device)
            tb = torch.tensor(Td_tr_n[b_idx], dtype=torch.float32, device=device)
            hb = torch.tensor(h_tr[b_idx], dtype=torch.long, device=device)
            wb = torch.tensor(sample_weights[b_idx], dtype=torch.float32, device=device)
            yb_true = torch.tensor(Ytr[b_idx], dtype=torch.float32, device=device)

            # Data augmentation: ruido gaussiano suave en GFS
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
                mb = mb + torch.randn_like(mb) * (noise_std * 0.5)

            pred_anom = model(xb, mb, tb, hb.float())
            y_pred = CLIM_T[hb] + pred_anom * ASD_T

            # Pérdida exacta alineada al Skill Score oficial (ponderada estacionalmente)
            err_sq = (y_pred - yb_true) ** 2  # (B, 70, 35, 4)
            per_sample_mse = err_sq.mean(dim=(1, 2))  # (B, 4)
            weighted_mse = (per_sample_mse * wb[:, None]).sum(dim=0) / (wb.sum() + 1e-8)  # (4,)
            rmse_per_var = torch.sqrt(weighted_mse + 1e-6)
            skill_loss = (rmse_per_var / CLIM_RMSE_TRN_T).mean()

            optimizer.zero_grad()
            skill_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, decay=0.992)

            running_loss += float(skill_loss.item())
            n_batches += 1

        # Evaluar pesos live y pesos EMA en validación temporal
        val_preds_live = predict_array(model, Wtr_n[val_mask], M24_tr_n[val_mask], Td_tr_n[val_mask], h_tr[val_mask])
        sc_live, rmse_live, skill_live = evaluate_skill_score(val_preds_live, Ytr[val_mask], clim_rmse_val)

        if ep >= 5:
            val_preds_ema = predict_array(ema_model, Wtr_n[val_mask], M24_tr_n[val_mask], Td_tr_n[val_mask], h_tr[val_mask])
            sc_ema, rmse_ema, skill_ema = evaluate_skill_score(val_preds_ema, Ytr[val_mask], clim_rmse_val)
        else:
            sc_ema, rmse_ema, skill_ema = -1e9, {}, {}

        if sc_ema >= sc_live:
            sc_ep, rmse_ep, skill_ep, src_ep = sc_ema, rmse_ema, skill_ema, "EMA"
            cand_state = ema_model.state_dict()
        else:
            sc_ep, rmse_ep, skill_ep, src_ep = sc_live, rmse_live, skill_live, "LIVE"
            cand_state = model.state_dict()

        dt = time.time() - t0
        marker = ""
        if sc_ep > best_score:
            best_score = sc_ep
            best_state = {k: v.detach().cpu().clone() for k, v in cand_state.items()}
            best_rmse_dict = rmse_ep
            best_skill_dict = skill_ep
            best_source = src_ep
            marker = f" ★ BEST ({src_ep})"

        print(
            f"Época {ep+1:02d}/{epochs:02d} · loss {running_loss/n_batches:.4f} · "
            f"Live {sc_live:.4f} | EMA {max(sc_ema, 0):.4f} · Skill/var: {skill_ep} · {dt:.1f}s{marker}"
        )

    print("=" * 80)
    print(f"🏆 MEJOR SKILL SCORE VALIDACIÓN (PhysicsUNet v3 · {best_source}): {best_score:.4f}")
    print(f"   Skill por variable : {best_skill_dict}")
    print(f"   RMSE por variable  : {best_rmse_dict}")
    print("=" * 80)

    model.load_state_dict(best_state)
    val_oof = predict_array(model, Wtr_n[val_mask], M24_tr_n[val_mask], Td_tr_n[val_mask], h_tr[val_mask])
    test_preds = predict_array(model, Wte_n, M24_te_n, Td_te_n, h_te)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    exp_id = logger.log_experiment(
        model_name="physics_unet_v3",
        metric_name="skill_score",
        fold_scores=[best_score],
        params={
            "arch": "physics_unet_v3",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "width": width,
            "dropout": dropout,
            "noise_std": noise_std,
            "n_params": n_params,
            "best_source": best_source,
            "val_rmse_per_var": best_rmse_dict,
            "val_skill_per_var": best_skill_dict,
        },
        features=[
            "GFS_centered_7lags_t-6..t+6",
            "GFS_24h_synoptic_mean",
            "DewPoint_Td_Magnus",
            "Orographic_Forcing_V_dot_gradDEM",
            "LapseRate_Prior",
            "DEM_10ch_Topography",
        ],
        oof_preds=val_oof,
        test_preds=test_preds,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=seed,
        notes=f"Skill={best_score:.4f} ({best_source}) | Skills={best_skill_dict}",
    )

    if best_score > prev_best_score:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = MODELS_DIR / "modelo.pt"
        torch.save(best_state, weights_path)
        size_mb = os.path.getsize(weights_path) / 1e6
        print(f"[Export] ¡NUEVO RÉCORD! Pesos guardados en {weights_path} ({size_mb:.2f} MB)")

        np.savez_compressed(
            MODELS_DIR / "norm_stats.npz",
            mu_x=mu_x,
            sd_x=sd_x,
            mu_td=np.array(mu_td, dtype=np.float32),
            sd_td=np.array(sd_td, dtype=np.float32),
            clim_trn=clim_trn,
            anom_sd=anom_sd,
            clim_rmse_val=clim_rmse_val,
            arch=np.array("physics_unet_v3"),
            width=np.array(width),
            best_score=np.array(best_score),
            exp_id=np.array(exp_id),
        )

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
        sub.to_csv(EXPERIMENTS_DIR / exp_id / "submission.csv", index=False)
        print(f"[Export] Nuevo submission.csv guardado en {sub_path} (shape={sub.shape})")
    else:
        print(f"[Info] No superó el récord previo ({prev_best_score:.4f}); se conservan los pesos previos en models/modelo.pt.")

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PhysicsUNet v3 — Hackathon IA PUCP 2026")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    train_v3(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        dropout=args.dropout,
        noise_std=args.noise_std,
        seed=args.seed,
    )
