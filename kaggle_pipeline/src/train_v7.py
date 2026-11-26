"""
Pipeline v7 — INTI-3K (Hackathon IA PUCP 2026 · Equipo PASSWORD).
Mejoras clave sobre v6 (0.3821–0.3835):
1. Ventana extendida con ciclo diurno ±24h: V7_LAGS = (-24, -6, -3, -1, 0, 1, 3, 6, 24).
2. Régimen sinóptico 24h: media móvil M24 y desviación estándar móvil S24 en [t-12h, t+12h].
3. Gradientes espaciales GFS (17x13, fila 0=Norte): dT/dx, dT/dy, dRH/dx, dRH/dy, Divergencia y Vorticidad.
4. Módulo espacial GFS2WRFCrossAttention (cada celda regional consulta tokens GFS de barlovento).
5. Suite de 4 arquitecturas compactas (~16 MB total) que devuelven predicción física directa (B, 70, 35, 4):
   - PhysicsConvNeXtV7    (ancla Climatología + CrossAttn)
   - DirectResConvNeXtV7  (ancla Física Directa GFS + Lapse-Rate + sesgo diurno aprendible, resistente a deriva estacional)
   - CrossAttnUNetV7      (U-Net 3 niveles + CrossAttn en cuello de botella)
   - CompactPhysicsFNOV7  (FNO compacto w=32, modes 12x8, regularizado para evitar overfitting)
6. Validación Holdout Temporal (>= 2024061600) + Validación Multi-Bloque Temporal (3 bloques de 14 días).
"""

import argparse
import copy
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from train_v3 import compute_dewpoint_c, update_ema

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

V7_LAGS = (-24, -6, -3, -1, 0, 1, 3, 6, 24)  # índice 4 = t, 2 = t-3, 6 = t+3, 0 = t-24, 8 = t+24


def build_v7_temporal_and_spatial_features(
    X: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye entradas en espacio GFS (17x13):
      - W    : (N, 9, 17, 13, 4) para V7_LAGS
      - M24  : (N, 17, 13, 4) media móvil [t-12h, t+12h]
      - S24  : (N, 17, 13, 4) std móvil [t-12h, t+12h]
      - GRAD : (N, 6, 17, 13) [dT_dx, dT_dy_north, dRH_dx, dRH_dy_north, div, vort]
      - TD   : (N, 1, 17, 13) punto de rocío (Magnus) en °C
    """
    ts = pd.to_datetime(t.astype(str), format="%Y%m%d%H")
    ts_ns = ts.values.astype("datetime64[h]").astype(np.int64)
    pos = {int(h): i for i, h in enumerate(ts_ns)}

    N = len(X)
    W = np.empty((N, len(V7_LAGS)) + X.shape[1:], dtype=np.float32)
    M24 = np.empty_like(X, dtype=np.float32)
    S24 = np.empty_like(X, dtype=np.float32)

    for i, cur_h in enumerate(ts_ns):
        for j, lag in enumerate(V7_LAGS):
            target_h = int(cur_h + lag)
            idx = pos.get(target_h, None)
            if idx is None:
                for delta in (1, -1, 2, -2, 3, -3):
                    idx = pos.get(target_h + delta, None)
                    if idx is not None:
                        break
            W[i, j] = X[idx] if idx is not None else X[i]

        win_idxs = [pos[int(cur_h + d)] for d in range(-12, 13) if int(cur_h + d) in pos]
        if len(win_idxs) >= 2:
            sub = X[win_idxs]
            M24[i] = sub.mean(axis=0)
            S24[i] = sub.std(axis=0)
        else:
            M24[i] = X[i]
            S24[i] = 0.0

    # Gradientes espaciales en GFS (17x13). OJO: fila 0 = Norte -> dy_north = -d_row
    rh = X[..., 0]
    temp = X[..., 1]
    u = X[..., 2]
    v = X[..., 3]

    dRH_row, dRH_dx = np.gradient(rh, axis=(1, 2))
    dT_row, dT_dx = np.gradient(temp, axis=(1, 2))
    dU_row, dU_dx = np.gradient(u, axis=(1, 2))
    dV_row, dV_dx = np.gradient(v, axis=(1, 2))

    dRH_dy = -dRH_row
    dT_dy = -dT_row
    dU_dy = -dU_row
    dV_dy = -dV_row

    div = dU_dx + dV_dy
    vort = dV_dx - dU_dy

    GRAD = np.stack([dT_dx, dT_dy, dRH_dx, dRH_dy, div, vort], axis=1).astype(np.float32)
    TD = compute_dewpoint_c(temp, rh)[:, None, :, :].astype(np.float32)
    return W, M24, S24, GRAD, TD


class GFS2WRFCrossAttention(nn.Module):
    """
    Atención espacial multi-cabeza desde la grilla regional WRF (18x9) hacia los tokens GFS (17x13),
    permitiendo consultar celdas de barlovento/sotavento más allá de la interpolación bilineal local.
    """

    def __init__(self, high_ch: int, gfs_ch: int, d_attn: int = 64, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Conv2d(high_ch, d_attn, kernel_size=1)
        self.kv_proj = nn.Conv2d(gfs_ch, d_attn, kernel_size=1)
        self.q_pos = nn.Parameter(0.02 * torch.randn(1, 18 * 9, d_attn))
        self.k_pos = nn.Parameter(0.02 * torch.randn(1, 17 * 13, d_attn))
        self.attn = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, dropout=0.05, batch_first=True)
        self.norm = nn.LayerNorm(d_attn)
        self.out_proj = nn.Conv2d(d_attn, high_ch, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.10))

    def forward(self, feat_high: torch.Tensor, feat_gfs: torch.Tensor) -> torch.Tensor:
        # feat_high: (B, high_ch, H, W), feat_gfs: (B, gfs_ch, 17, 13)
        B, _, H, W = feat_high.shape
        q_map = F.adaptive_avg_pool2d(self.q_proj(feat_high), (18, 9))  # (B, d, 18, 9)
        q_seq = q_map.flatten(2).transpose(1, 2) + self.q_pos  # (B, 162, d)

        k_map = self.kv_proj(feat_gfs)  # (B, d, 17, 13)
        k_seq = k_map.flatten(2).transpose(1, 2) + self.k_pos  # (B, 221, d)

        attn_out, _ = self.attn(q_seq, k_seq, k_seq, need_weights=False)
        attn_out = self.norm(q_seq + attn_out)
        attn_map = attn_out.transpose(1, 2).reshape(B, -1, 18, 9)
        attn_up = F.interpolate(attn_map, size=(H, W), mode="bilinear", align_corners=True)
        return feat_high + self.alpha * self.out_proj(attn_up)


def _assemble_v7_gfs_tensor(
    w_norm: torch.Tensor,  # (B, 9, 17, 13, 4)
    m24_norm: torch.Tensor,  # (B, 17, 13, 4)
    s24_norm: torch.Tensor,  # (B, 17, 13, 4)
    grad_norm: torch.Tensor,  # (B, 6, 17, 13)
    td_norm: torch.Tensor,  # (B, 1, 17, 13)
    h: torch.Tensor,  # (B,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construye los 74 canales de entrada en 17x13 y devuelve (gfs_in, xt, h_vec)."""
    B = w_norm.shape[0]
    xt = w_norm[:, 4]  # t
    xm3 = w_norm[:, 2]  # t-3
    xp3 = w_norm[:, 6]  # t+3
    xm24 = w_norm[:, 0]  # t-24

    wspd = torch.stack(
        [torch.sqrt(w_norm[:, idx, :, :, 2] ** 2 + w_norm[:, idx, :, :, 3] ** 2 + 1e-6) for idx in (0, 2, 4, 6, 8)],
        dim=1,
    )  # (B, 5, 17, 13)

    d_prev = (xt - xm3).permute(0, 3, 1, 2)
    d_next = (xp3 - xt).permute(0, 3, 1, 2)
    d_diurnal = (xt - m24_norm).permute(0, 3, 1, 2)
    d_24h = (xt - xm24).permute(0, 3, 1, 2)

    w_flat = w_norm.permute(0, 1, 4, 2, 3).reshape(B, 36, 17, 13)
    m24_flat = m24_norm.permute(0, 3, 1, 2)
    s24_flat = s24_norm.permute(0, 3, 1, 2)

    ang = 2.0 * np.pi * h / 24.0
    h_vec = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
    hh_low = h_vec[..., None, None].expand(B, 2, 17, 13)

    # 36 + 4 + 4 + 6 + 1 + 5 + 16 + 2 = 74 canales
    gfs_in = torch.cat(
        [w_flat, m24_flat, s24_flat, grad_norm, td_norm, wspd, d_prev, d_next, d_diurnal, d_24h, hh_low],
        dim=1,
    )
    return gfs_in, xt, h_vec


def _fuse_highres_v7(
    f_low: torch.Tensor,
    xt: torch.Tensor,
    m24_norm: torch.Tensor,
    s24_norm: torch.Tensor,
    grad_norm: torch.Tensor,
    td_norm: torch.Tensor,
    h_vec: torch.Tensor,
    grid: torch.Tensor,
    topo: torch.Tensor,
    lapse_delta: torch.Tensor,
    learned_static: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = xt.shape[0]
    grid_b = grid.expand(B, -1, -1, -1)
    f_up = F.grid_sample(f_low, grid_b, mode="bilinear", align_corners=True)
    raw_cat = torch.cat(
        [xt.permute(0, 3, 1, 2), m24_norm.permute(0, 3, 1, 2), s24_norm.permute(0, 3, 1, 2), grad_norm, td_norm],
        dim=1,
    )  # 4 + 4 + 4 + 6 + 1 = 19 ch
    raw_up = F.grid_sample(raw_cat, grid_b, mode="bilinear", align_corners=True)

    topo_b = topo.expand(B, -1, -1, -1)
    u_up, v_up = raw_up[:, 2:3], raw_up[:, 3:4]
    dy_n, dx_n = topo_b[:, 1:2], topo_b[:, 2:3]
    oro_feats = torch.cat([u_up * dx_n + v_up * dy_n, u_up * dy_n - v_up * dx_n], dim=1)

    t_prior = raw_up[:, 1:2] + lapse_delta.expand(B, -1, -1, -1)
    static_b = learned_static.expand(B, -1, -1, -1)
    hh_high = h_vec[..., None, None].expand(B, 2, 70, 35)

    fused = torch.cat([f_up, raw_up, t_prior, oro_feats, topo_b, static_b, hh_high], dim=1)
    return fused, topo_b, oro_feats


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1, expand: int = 3, dropout: float = 0.14):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expand, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pw2 = nn.Conv2d(dim * expand, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-3 * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pw2(self.drop(self.act(self.pw1(self.norm(self.dwconv(x))))))


class CompactResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.12):
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


class PhysicsConvNeXtV7(nn.Module):
    """Familia 1: ConvNeXt 7x7 dilatado + Cross-Attention GFS->WRF, ancla en Climatología."""

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        clim_trn: torch.Tensor,
        anom_sd: torch.Tensor,
        w: int = 80,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])
        self.register_buffer("clim_trn", clim_trn)
        self.register_buffer("anom_sd", anom_sd)

        self.gfs_stem = nn.Sequential(
            nn.Conv2d(74, w, 3, padding=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        fused_ch = w + 19 + 1 + 2 + 10 + 12 + 2
        self.highres_stem = nn.Sequential(nn.Conv2d(fused_ch, w, 1), nn.GroupNorm(1, w), nn.GELU())
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w, gfs_ch=w, d_attn=64, num_heads=4)
        self.blocks = nn.Sequential(
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 2, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 3, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        head_in = w + 12
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_in, xt, h_vec = _assemble_v7_gfs_tensor(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h.float())
        f_low = self.gfs_stem(gfs_in)
        fused, topo_b, oro_feats = _fuse_highres_v7(
            f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec,
            self.grid, self.topo, self.lapse_delta, self.learned_static,
        )
        feat = self.cross_attn(self.highres_stem(fused), f_low)
        feat = self.blocks(feat)
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        anom = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.clim_trn[h] + anom * self.anom_sd


class DirectResConvNeXtV7(nn.Module):
    """
    Familia 2: ConvNeXt 7x7 con Ancla Física Directa sobre GFS interpolado + Lapse-Rate + sesgo diurno.
    No depende de la media húmeda feb-abr de la climatología, mejorando la generalización estacional.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        lapse_phys_t: torch.Tensor,  # (70, 35) en Kelvin
        mu_x: torch.Tensor,  # (4,)
        sd_x: torch.Tensor,  # (4,)
        anom_sd: torch.Tensor,  # (4,)
        w: int = 80,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])
        self.register_buffer("lapse_phys_t", lapse_phys_t[None, :, :, None])  # (1, 70, 35, 1)
        self.register_buffer("mu_x", mu_x)
        self.register_buffer("sd_x", sd_x)
        self.register_buffer("anom_sd", anom_sd)

        self.gfs_stem = nn.Sequential(
            nn.Conv2d(74, w, 3, padding=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        # Sesgo diurno por hora UTC inicializado en cero
        self.diurnal_bias = nn.Parameter(torch.zeros(24, 70, 35, 4))

        fused_ch = w + 19 + 1 + 2 + 10 + 12 + 2
        self.highres_stem = nn.Sequential(nn.Conv2d(fused_ch, w, 1), nn.GroupNorm(1, w), nn.GELU())
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w, gfs_ch=w, d_attn=64, num_heads=4)
        self.blocks = nn.Sequential(
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 2, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 3, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        head_in = w + 12
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        B = w_norm.shape[0]
        gfs_in, xt, h_vec = _assemble_v7_gfs_tensor(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h.float())
        f_low = self.gfs_stem(gfs_in)
        fused, topo_b, oro_feats = _fuse_highres_v7(
            f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec,
            self.grid, self.topo, self.lapse_delta, self.learned_static,
        )
        feat = self.cross_attn(self.highres_stem(fused), f_low)
        feat = self.blocks(feat)
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        res = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)

        # Ancla física directa en unidades físicas (RH, T_K, U, V)
        xt_phys = xt * self.sd_x + self.mu_x  # (B, 17, 13, 4)
        grid_b = self.grid.expand(B, -1, -1, -1)
        anchor = F.grid_sample(xt_phys.permute(0, 3, 1, 2), grid_b, mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
        # Sumar ajuste por lapse-rate únicamente al canal de Temperatura (índice 1)
        anchor_t = anchor[..., 1:2] + self.lapse_phys_t
        anchor = torch.cat([anchor[..., 0:1], anchor_t, anchor[..., 2:4]], dim=-1)

        return anchor + self.diurnal_bias[h] + res * self.anom_sd


class CrossAttnUNetV7(nn.Module):
    """Familia 3: U-Net 3 niveles con Cross-Attention GFS->WRF en el cuello de botella 18x9."""

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        clim_trn: torch.Tensor,
        anom_sd: torch.Tensor,
        w: int = 64,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])
        self.register_buffer("clim_trn", clim_trn)
        self.register_buffer("anom_sd", anom_sd)

        self.gfs_enc = nn.Sequential(
            nn.Conv2d(74, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            CompactResBlock(w, dropout=dropout * 0.5),
            CompactResBlock(w, dropout=dropout * 0.5),
        )
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        fused_ch = w + 19 + 1 + 2 + 10 + 12 + 2
        self.enc1 = nn.Sequential(nn.Conv2d(fused_ch, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(), CompactResBlock(w, dropout))
        self.enc2 = nn.Sequential(nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout))
        self.enc3 = nn.Sequential(nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout * 1.25))
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w * 2, gfs_ch=w, d_attn=64, num_heads=4)
        self.dec2 = nn.Sequential(nn.Conv2d(w * 4, w * 2, 3, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout))
        self.dec1 = nn.Sequential(nn.Conv2d(w * 3, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(), CompactResBlock(w, dropout * 0.5))

        head_in = w + 12
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_in, xt, h_vec = _assemble_v7_gfs_tensor(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h.float())
        f_low = self.gfs_enc(gfs_in)
        fused, topo_b, oro_feats = _fuse_highres_v7(
            f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec,
            self.grid, self.topo, self.lapse_delta, self.learned_static,
        )
        s1 = self.enc1(fused)
        s2 = self.enc2(s1)
        s3 = self.cross_attn(self.enc3(s2), f_low)
        u2 = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([s2, u2], dim=1))
        u1 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([s1, u1], dim=1))
        head_in = torch.cat([d1, topo_b, oro_feats], dim=1)
        anom = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.clim_trn[h] + anom * self.anom_sd


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 8):
        super().__init__()
        self.out_channels, self.modes1, self.modes2 = out_channels, modes1, modes2
        scale = 1.0 / (in_channels * out_channels)
        self.w1_r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w1_i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w2_r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w2_i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    @staticmethod
    def _compl_mul2d(xr, xi, wr, wi):
        return torch.complex(
            torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi),
            torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        x1, x2 = x_ft[:, :, :m1, :m2], x_ft[:, :, -m1:, :m2]
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x1.real, x1.imag, self.w1_r[:, :, :m1, :m2], self.w1_i[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x2.real, x2.imag, self.w2_r[:, :, :m1, :m2], self.w2_i[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock2D(nn.Module):
    def __init__(self, dim: int, modes1: int = 12, modes2: int = 8, dropout: float = 0.18):
        super().__init__()
        self.spectral = SpectralConv2d(dim, dim, modes1=modes1, modes2=modes2)
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop(self.act(self.norm(self.spectral(x) + self.pw(self.local_conv(x)))))


class CompactPhysicsFNOV7(nn.Module):
    """Familia 4: FNO compacto y altamente regularizado (w=32, modes 12x8, ~3 MB vs 20.6 MB previos)."""

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        clim_trn: torch.Tensor,
        anom_sd: torch.Tensor,
        w: int = 32,
        dropout: float = 0.18,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])
        self.register_buffer("clim_trn", clim_trn)
        self.register_buffer("anom_sd", anom_sd)

        self.gfs_stem = nn.Sequential(
            nn.Conv2d(74, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            FNOBlock2D(w, modes1=5, modes2=4, dropout=dropout * 0.5),
        )
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)
        fused_ch = w + 19 + 1 + 2 + 10 + 12 + 2
        self.lift = nn.Sequential(nn.Conv2d(fused_ch, w, 1, bias=False), nn.GroupNorm(8, w), nn.GELU())
        self.fno_layers = nn.Sequential(
            FNOBlock2D(w, 12, 8, dropout),
            FNOBlock2D(w, 12, 8, dropout),
            FNOBlock2D(w, 8, 6, dropout),
        )
        head_in = w + 12
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 2, 1))
        self.head_wind = nn.Sequential(nn.Conv2d(head_in, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 2, 1))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_in, xt, h_vec = _assemble_v7_gfs_tensor(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h.float())
        f_low = self.gfs_stem(gfs_in)
        fused, topo_b, oro_feats = _fuse_highres_v7(
            f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec,
            self.grid, self.topo, self.lapse_delta, self.learned_static,
        )
        feat = self.fno_layers(self.lift(fused))
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        anom = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.clim_trn[h] + anom * self.anom_sd


class EnsembleV7(nn.Module):
    """Módulo único que combina las 4 familias v7 en espacio físico (B, 70, 35, 4)."""

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        lapse_phys_t: torch.Tensor,
        mu_x: torch.Tensor,
        sd_x: torch.Tensor,
        clim_trn: torch.Tensor,
        anom_sd: torch.Tensor,
        blend_weights: np.ndarray,
    ):
        super().__init__()
        self.m_conv = PhysicsConvNeXtV7(grid, topo_features, lapse_delta_norm, clim_trn, anom_sd, w=80, dropout=0.0)
        self.m_direct = DirectResConvNeXtV7(grid, topo_features, lapse_delta_norm, lapse_phys_t, mu_x, sd_x, anom_sd, w=80, dropout=0.0)
        self.m_unet = CrossAttnUNetV7(grid, topo_features, lapse_delta_norm, clim_trn, anom_sd, w=64, dropout=0.0)
        self.m_fno = CompactPhysicsFNOV7(grid, topo_features, lapse_delta_norm, clim_trn, anom_sd, w=32, dropout=0.0)
        self.register_buffer("blend_weights", torch.tensor(blend_weights, dtype=torch.float32))  # (4, 4)

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        y0 = self.m_conv(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y1 = self.m_direct(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y2 = self.m_unet(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y3 = self.m_fno(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        bw = self.blend_weights
        return (
            bw[0].view(1, 1, 1, 4) * y0
            + bw[1].view(1, 1, 1, 4) * y1
            + bw[2].view(1, 1, 1, 4) * y2
            + bw[3].view(1, 1, 1, 4) * y3
        )


def predict_v7(net: nn.Module, Wn, Mn, Sn, Gn, Tn, h_all, device: str, bs: int = 128) -> np.ndarray:
    net.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(Wn), bs):
            xb = torch.tensor(Wn[s : s + bs], dtype=torch.float32, device=device)
            mb = torch.tensor(Mn[s : s + bs], dtype=torch.float32, device=device)
            sb = torch.tensor(Sn[s : s + bs], dtype=torch.float32, device=device)
            gb = torch.tensor(Gn[s : s + bs], dtype=torch.float32, device=device)
            tb = torch.tensor(Tn[s : s + bs], dtype=torch.float32, device=device)
            hb = torch.tensor(h_all[s : s + bs], dtype=torch.long, device=device)
            y_hat = net(xb, mb, sb, gb, tb, hb).cpu().numpy()
            y_hat[..., 0] = np.clip(y_hat[..., 0], 0.0, 100.0)
            preds.append(y_hat)
    return np.concatenate(preds, axis=0)


def train_member_v7(
    name: str,
    model: nn.Module,
    trn_mask: np.ndarray,
    val_mask: np.ndarray,
    Wn: np.ndarray,
    Mn: np.ndarray,
    Sn: np.ndarray,
    Gn: np.ndarray,
    Tn: np.ndarray,
    h_tr: np.ndarray,
    Ytr: np.ndarray,
    sample_weights: np.ndarray,
    clim_rmse_trn_t: torch.Tensor,
    clim_rmse_val: np.ndarray,
    device: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1.8e-3,
    weight_decay: float = 1.5e-2,
    noise_std: float = 0.045,
    seed: int = 42,
):
    seed_everything(seed)
    model = model.to(device)
    ema_model = copy.deepcopy(model).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{name}] Parámetros entrenables: {n_params:,} | lr={lr} | seed={seed}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    itr = np.where(trn_mask)[0]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * int(np.ceil(len(itr) / batch_size)), pct_start=0.15
    )

    best_sc, best_state, best_src, best_ep = -1e9, None, "LIVE", 1
    best_skill_d, best_rmse_d = {}, {}
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        np.random.shuffle(itr)
        running_loss, n_batches = 0.0, 0
        for s in range(0, len(itr), batch_size):
            b = itr[s : s + batch_size]
            xb = torch.tensor(Wn[b], dtype=torch.float32, device=device)
            mb = torch.tensor(Mn[b], dtype=torch.float32, device=device)
            sb = torch.tensor(Sn[b], dtype=torch.float32, device=device)
            gb = torch.tensor(Gn[b], dtype=torch.float32, device=device)
            tb = torch.tensor(Tn[b], dtype=torch.float32, device=device)
            hb = torch.tensor(h_tr[b], dtype=torch.long, device=device)
            wb = torch.tensor(sample_weights[b], dtype=torch.float32, device=device)
            yb = torch.tensor(Ytr[b], dtype=torch.float32, device=device)

            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
                mb = mb + torch.randn_like(mb) * (noise_std * 0.5)

            y_pred = model(xb, mb, sb, gb, tb, hb)
            err_sq = ((y_pred - yb) ** 2).mean(dim=(1, 2))
            w_mse = (err_sq * wb[:, None]).sum(dim=0) / (wb.sum() + 1e-8)
            skill_loss = (torch.sqrt(w_mse + 1e-6) / clim_rmse_trn_t).mean()

            optimizer.zero_grad()
            skill_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, decay=0.992)

            running_loss += float(skill_loss.item())
            n_batches += 1

        vp_live = predict_v7(model, Wn[val_mask], Mn[val_mask], Sn[val_mask], Gn[val_mask], Tn[val_mask], h_tr[val_mask], device)
        sc_live, rmse_live, skill_live = evaluate_skill_score(vp_live, Ytr[val_mask], clim_rmse_val)
        sc_ema, rmse_ema, skill_ema = -1e9, {}, {}
        if ep >= 5:
            vp_ema = predict_v7(ema_model, Wn[val_mask], Mn[val_mask], Sn[val_mask], Gn[val_mask], Tn[val_mask], h_tr[val_mask], device)
            sc_ema, rmse_ema, skill_ema = evaluate_skill_score(vp_ema, Ytr[val_mask], clim_rmse_val)

        if sc_ema >= sc_live:
            sc_ep, rmse_ep, skill_ep, src_ep, cand = sc_ema, rmse_ema, skill_ema, "EMA", ema_model
        else:
            sc_ep, rmse_ep, skill_ep, src_ep, cand = sc_live, rmse_live, skill_live, "LIVE", model

        mark = ""
        if sc_ep > best_sc:
            best_sc = sc_ep
            best_src = src_ep
            best_ep = ep + 1
            best_rmse_d = rmse_ep
            best_skill_d = skill_ep
            best_state = {k: v.detach().cpu().clone() for k, v in cand.state_dict().items()}
            mark = f" ★ BEST ({src_ep})"

        if (ep + 1) % 5 == 0 or mark:
            print(
                f"  Época {ep+1:02d}/{epochs} · loss {running_loss/n_batches:.4f} · "
                f"Live {sc_live:.4f} | EMA {max(sc_ema, 0):.4f} · {skill_ep}{mark}"
            )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    val_oof = predict_v7(model, Wn[val_mask], Mn[val_mask], Sn[val_mask], Gn[val_mask], Tn[val_mask], h_tr[val_mask], device)
    sc_fin, rmse_fin, skill_fin = evaluate_skill_score(val_oof, Ytr[val_mask], clim_rmse_val)
    print(f"--> {name} listo en {time.time()-t_start:.1f}s | Best Ep={best_ep} ({best_src}) | Skill={sc_fin:.4f} | {skill_fin}")

    del model, ema_model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return best_state, val_oof, sc_fin, best_ep, skill_fin, rmse_fin


def prepare_split_tensors(Xtr, Ytr, ttr, st, trn_mask, val_mask, device: str):
    dem = st["dem"].astype(np.float32)
    h_tr = extract_hour_utc(ttr)
    clim_trn = compute_climatology(Ytr[trn_mask], ttr[trn_mask])
    clim_rmse_val = np.sqrt(((clim_trn[h_tr[val_mask]] - Ytr[val_mask]) ** 2).mean(axis=(0, 1, 2)))
    clim_rmse_trn = np.sqrt(((clim_trn[h_tr[trn_mask]] - Ytr[trn_mask]) ** 2).mean(axis=(0, 1, 2))).astype(np.float32)

    W, M24, S24, GRAD, TD = build_v7_temporal_and_spatial_features(Xtr, ttr)

    mu_x = Xtr[trn_mask].mean(axis=(0, 1, 2)).astype(np.float32)
    sd_x = Xtr[trn_mask].std(axis=(0, 1, 2)).astype(np.float32)
    mu_td = float(TD[trn_mask].mean())
    sd_td = float(TD[trn_mask].std() + 1e-6)
    mu_grad = GRAD[trn_mask].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)  # (1, 6, 1, 1)
    sd_grad = (GRAD[trn_mask].std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
    anom_sd = (Ytr[trn_mask] - clim_trn[h_tr[trn_mask]]).std(axis=(0, 1, 2)).astype(np.float32)

    Wn = ((W - mu_x) / sd_x).astype(np.float32)
    Mn = ((M24 - mu_x) / sd_x).astype(np.float32)
    Sn = (S24 / sd_x).astype(np.float32)
    Gn = ((GRAD - mu_grad) / sd_grad).astype(np.float32)
    Tn = ((TD - mu_td) / sd_td).astype(np.float32)

    months_tr = pd.to_datetime(ttr, format="%Y%m%d%H").month.values
    month_w_map = {2: 0.50, 3: 0.80, 4: 0.90, 5: 1.15, 6: 1.45}
    sample_weights = np.array([month_w_map.get(int(m), 1.0) for m in months_tr], dtype=np.float32)
    sample_weights[trn_mask] /= sample_weights[trn_mask].mean()

    grid_t = build_sampling_grid(st, device)
    topo_np = build_topographic_features(dem, st["wrf_lats"], st["wrf_lons"])
    topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
    dem_smooth = smooth_2d(dem, k=9)
    lapse_phys = (-0.0065 * (dem - dem_smooth)).astype(np.float32)
    lapse_delta_norm = torch.tensor(lapse_phys / float(sd_x[1]), dtype=torch.float32, device=device)
    lapse_phys_t = torch.tensor(lapse_phys, dtype=torch.float32, device=device)

    return {
        "h_tr": h_tr,
        "clim_trn": clim_trn,
        "clim_rmse_val": clim_rmse_val,
        "clim_rmse_trn_t": torch.tensor(clim_rmse_trn, dtype=torch.float32, device=device),
        "clim_trn_t": torch.tensor(clim_trn, dtype=torch.float32, device=device),
        "anom_sd_t": torch.tensor(anom_sd, dtype=torch.float32, device=device),
        "mu_x_t": torch.tensor(mu_x, dtype=torch.float32, device=device),
        "sd_x_t": torch.tensor(sd_x, dtype=torch.float32, device=device),
        "Wn": Wn,
        "Mn": Mn,
        "Sn": Sn,
        "Gn": Gn,
        "Tn": Tn,
        "sample_weights": sample_weights,
        "grid_t": grid_t,
        "topo_t": topo_t,
        "lapse_delta_norm": lapse_delta_norm,
        "lapse_phys_t": lapse_phys_t,
    }


def run_multiblock_validation(Xtr, Ytr, ttr, st, device: str, epochs: int = 24) -> None:
    """Validación Temporal Multi-Bloque (3 bloques de 14 días en Mayo–Junio) para comprobar estabilidad."""
    print("\n" + "=" * 80)
    print("VALIDACIÓN TEMPORAL MULTI-BLOQUE (3 BLOQUES DE 14 DÍAS: MAYO–JUNIO)")
    print("=" * 80)
    blocks = [
        ("Bloque 1 (19 May - 01 Jun)", "2024051900", "2024060123"),
        ("Bloque 2 (02 Jun - 15 Jun)", "2024060200", "2024061523"),
        ("Bloque 3 (16 Jun - 30 Jun)", "2024061600", "2024063023"),
    ]
    scores_conv, scores_dir = [], []
    for lbl, start_t, end_t in blocks:
        val_m = (ttr >= start_t) & (ttr <= end_t)
        trn_m = ttr < start_t  # estricto causal hacia adelante
        d = prepare_split_tensors(Xtr, Ytr, ttr, st, trn_m, val_m, device)
        _, _, sc_c, ep_c, _, _ = train_member_v7(
            f"MultiBlock-ConvNeXt ({lbl})",
            PhysicsConvNeXtV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["clim_trn_t"], d["anom_sd_t"], w=80, dropout=0.15),
            trn_m, val_m, d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
            d["sample_weights"], d["clim_rmse_trn_t"], d["clim_rmse_val"], device,
            epochs=epochs, lr=1.8e-3, weight_decay=1.5e-2, noise_std=0.045, seed=42,
        )
        _, _, sc_d, ep_d, _, _ = train_member_v7(
            f"MultiBlock-DirectRes ({lbl})",
            DirectResConvNeXtV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["lapse_phys_t"], d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=80, dropout=0.15),
            trn_m, val_m, d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
            d["sample_weights"], d["clim_rmse_trn_t"], d["clim_rmse_val"], device,
            epochs=epochs, lr=1.8e-3, weight_decay=1.5e-2, noise_std=0.045, seed=102,
        )
        scores_conv.append((sc_c, ep_c))
        scores_dir.append((sc_d, ep_d))

    print("\n--- Resumen Multi-Bloque ---")
    for (lbl, _, _), (sc_c, ep_c), (sc_d, ep_d) in zip(blocks, scores_conv, scores_dir):
        print(f"  {lbl}: ConvNeXt={sc_c:.4f} (ep {ep_c}) | DirectRes={sc_d:.4f} (ep {ep_d})")
    print(f"  Media Multi-Bloque -> ConvNeXt={np.mean([s for s,_ in scores_conv]):.4f} | DirectRes={np.mean([s for s,_ in scores_dir]):.4f}")


def main(epochs: int = 30, run_multiblock: bool = True) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train v7] Dispositivo: {device.upper()} | Épocas Holdout: {epochs}")

    tr = np.load(DATA_DIR / "train.npz")
    st = dict(np.load(DATA_DIR / "static.npz"))
    Xtr, Ytr, ttr = tr["X"], tr["Y"], tr["times"].astype(str)

    val_mask = ttr >= "2024061600"
    trn_mask = ~val_mask

    d = prepare_split_tensors(Xtr, Ytr, ttr, st, trn_mask, val_mask, device)
    y_val = Ytr[val_mask]
    clim_rmse_val = d["clim_rmse_val"]

    specs = [
        (
            "PhysicsConvNeXtV7",
            PhysicsConvNeXtV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["clim_trn_t"], d["anom_sd_t"], w=80, dropout=0.15),
            1.8e-3, 1.5e-2, 0.045, 42,
        ),
        (
            "DirectResConvNeXtV7",
            DirectResConvNeXtV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["lapse_phys_t"], d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=80, dropout=0.15),
            1.8e-3, 1.5e-2, 0.045, 102,
        ),
        (
            "CrossAttnUNetV7",
            CrossAttnUNetV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["clim_trn_t"], d["anom_sd_t"], w=64, dropout=0.12),
            1.5e-3, 1.2e-2, 0.040, 203,
        ),
        (
            "CompactPhysicsFNOV7",
            CompactPhysicsFNOV7(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], d["clim_trn_t"], d["anom_sd_t"], w=32, dropout=0.18),
            1.8e-3, 3.0e-2, 0.050, 304,
        ),
    ]

    oofs: List[np.ndarray] = []
    member_results = []
    for name, model, lr, wd, n_std, seed in specs:
        _, oof, sc, best_ep, skill_d, rmse_d = train_member_v7(
            name, model, trn_mask, val_mask,
            d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
            d["sample_weights"], d["clim_rmse_trn_t"], clim_rmse_val, device,
            epochs=epochs, lr=lr, weight_decay=wd, noise_std=n_std, seed=seed,
        )
        oofs.append(oof)
        member_results.append((name, sc, best_ep, skill_d, rmse_d))

    # Evaluar promedio uniforme vs pesos regularizados por variable (pasos 0.05, penalidad suave hacia 0.25)
    oof_eq = sum(oofs) / len(oofs)
    sc_eq, rmse_eq, skill_eq = evaluate_skill_score(oof_eq, y_val, clim_rmse_val)

    # Búsqueda en simplex de 4 miembros con pasos de 0.10 + refinamiento suave (evita overfitting)
    grid_w = np.linspace(0.0, 1.0, 11)
    blend_w = np.zeros((4, 4), dtype=np.float32)
    oof_opt = np.empty_like(oof_eq)
    for v_idx in range(4):
        best_obj = 1e9
        best_wv = (0.25, 0.25, 0.25, 0.25)
        for w0 in grid_w:
            for w1 in grid_w:
                if w0 + w1 > 1.001:
                    continue
                for w2 in grid_w:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-6:
                        continue
                    w3 = max(0.0, w3)
                    pred_v = w0 * oofs[0][..., v_idx] + w1 * oofs[1][..., v_idx] + w2 * oofs[2][..., v_idx] + w3 * oofs[3][..., v_idx]
                    rmse_v = float(np.sqrt(((pred_v - y_val[..., v_idx]) ** 2).mean()))
                    # Regularización muy suave hacia uniforme (0.25 c/u)
                    reg = 0.002 * ((w0 - 0.25) ** 2 + (w1 - 0.25) ** 2 + (w2 - 0.25) ** 2 + (w3 - 0.25) ** 2)
                    obj = rmse_v / clim_rmse_val[v_idx] + reg
                    if obj < best_obj:
                        best_obj = obj
                        best_wv = (round(float(w0), 2), round(float(w1), 2), round(float(w2), 2), round(float(w3), 2))
        blend_w[:, v_idx] = best_wv
        oof_opt[..., v_idx] = sum(best_wv[k] * oofs[k][..., v_idx] for k in range(4))

    sc_opt, rmse_opt, skill_opt = evaluate_skill_score(oof_opt, y_val, clim_rmse_val)

    print("\n" + "=" * 85)
    print("RESUMEN FASE A — HOLDOUT TEMPORAL (>= 2024061600) — SUITE v7")
    print("=" * 85)
    for name, sc, ep, sk, rm in member_results:
        print(f"  {name:22s} | Ep={ep:02d} | Skill={sc:.4f} | Var={sk}")
    print(f"  {'Ensamble Uniforme 25%':22s} |          | Skill={sc_eq:.4f} | Var={skill_eq}")
    print(f"  {'Ensamble Óptimo Reg.':22s} |          | Skill={sc_opt:.4f} | Var={skill_opt}")
    print(f"  RMSE Ensamble Óptimo   : {rmse_opt}")
    print(f"  Pesos (Conv, Dir, UNet, FNO) por [RH, T, U, V]:\n{blend_w}")
    print("=" * 85)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    logger.log_experiment(
        model_name="ensemble_v7_holdout",
        metric_name="skill_score",
        fold_scores=[sc_opt],
        params={
            "members": [{"name": n, "skill": round(s, 4), "best_ep": ep} for n, s, ep, _, _ in member_results],
            "blend_weights": blend_w.tolist(),
            "uniform_skill": round(sc_eq, 4),
            "val_rmse_per_var": rmse_opt,
            "val_skill_per_var": skill_opt,
        },
        features=["V7_LAGS_+-24h", "M24_S24", "GFS_Div_Vort_dT_dRH", "GFS2WRF_CrossAttn", "DirectRes_Anchor"],
        oof_preds=oof_opt,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=42,
        notes=f"Skill={sc_opt:.4f} (eq={sc_eq:.4f}) | {skill_opt}",
    )

    if run_multiblock:
        run_multiblock_validation(Xtr, Ytr, ttr, st, device, epochs=24)

    return sc_opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación v7 — Hackathon IA PUCP 2026")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-multiblock", action="store_true")
    args = parser.parse_args()
    main(epochs=args.epochs, run_multiblock=not args.no_multiblock)
