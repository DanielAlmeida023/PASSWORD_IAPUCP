"""
Pipeline v8 — INTI-3K (Hackathon IA PUCP 2026 · Equipo PASSWORD).
Mejoras clave sobre v7 (0.3984–0.3988 holdout):
1. TemporalLagMixer (atención temporal multi-cabeza sobre los 9 lags ±24h por celda GFS).
2. Ancla Física Directa (DirectPhysicalAnchor) en TODOS los miembros de la suite
   (cero ancla a la climatología húmeda feb-abr), con:
   - Gradiente térmico altitudinal aprendible por hora UTC (lapse_rate_h, K/km)
   - Pendiente altitudinal de humedad relativa aprendible por hora UTC (rh_elev_h, %/km)
   - Sesgo diurno microclimático por píxel de 3 km (diurnal_bias)
3. Prior físico adicional en 3 km: rh_prior_norm (RH adiabática vía Magnus-Tetens segura).
4. Suite de 4 arquitecturas diversas 100% DirectRes (~16-20 MB):
   - DirectResConvNeXtV8
   - DirectResMultiScaleV8
   - DirectResUNetV8
   - DirectResFNOV8
"""

import argparse
import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_logger import ExperimentLogger
from train import evaluate_skill_score, smooth_2d
from train_v7 import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    VARS,
    CompactResBlock,
    ConvNeXtBlock2D,
    FNOBlock2D,
    GFS2WRFCrossAttention,
    _assemble_v7_gfs_tensor,
    predict_v7,
    prepare_split_tensors,
    train_member_v7,
)


class TemporalLagMixer(nn.Module):
    """Atención temporal multi-cabeza sobre los 9 lags [-24..+24] por celda GFS (17x13)."""

    def __init__(self, d_t: int = 24, num_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Linear(4, d_t)
        self.lag_pos = nn.Parameter(0.02 * torch.randn(1, 9, d_t))
        self.attn = nn.MultiheadAttention(embed_dim=d_t, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_t)
        self.out_proj = nn.Sequential(nn.Linear(d_t * 2, d_t), nn.GELU())

    def forward(self, w_norm: torch.Tensor) -> torch.Tensor:
        # w_norm: (B, 9, 17, 13, 4) -> tokens por celda (B*221, 9, d_t)
        B, L, H, W, C = w_norm.shape
        tokens = w_norm.permute(0, 2, 3, 1, 4).reshape(B * H * W, L, C)
        x = self.in_proj(tokens) + self.lag_pos
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        # Combina instante central (idx=4 -> t) y resumen global de los 9 lags
        pooled = torch.cat([x[:, 4], x.mean(dim=1)], dim=-1)
        out = self.out_proj(pooled).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class DirectPhysicalAnchor(nn.Module):
    """Ancla física directa GFS + lapse-rate térmico/hídrico aprendible por hora UTC + sesgo diurno."""

    def __init__(
        self,
        grid: torch.Tensor,
        dem_delta_km: torch.Tensor,  # (70, 35)
        mu_x: torch.Tensor,
        sd_x: torch.Tensor,
        anom_sd: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("dem_delta_km", dem_delta_km[None, :, :, None])  # (1, 70, 35, 1)
        self.register_buffer("mu_x", mu_x)
        self.register_buffer("sd_x", sd_x)
        self.register_buffer("anom_sd", anom_sd)

        self.lapse_rate_h = nn.Parameter(torch.full((24, 1, 1, 1), -5.0))
        self.rh_elev_h = nn.Parameter(torch.zeros(24, 1, 1, 1))
        self.diurnal_bias = nn.Parameter(torch.zeros(24, 70, 35, 4))

    def forward(self, xt: torch.Tensor, h: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        B = xt.shape[0]
        xt_phys = xt * self.sd_x + self.mu_x
        anchor = F.grid_sample(
            xt_phys.permute(0, 3, 1, 2),
            self.grid.expand(B, -1, -1, -1),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        rh_anchor = anchor[..., 0:1] + self.rh_elev_h[h] * self.dem_delta_km
        t_anchor = anchor[..., 1:2] + self.lapse_rate_h[h] * self.dem_delta_km
        anchor_adj = torch.cat([rh_anchor, t_anchor, anchor[..., 2:4]], dim=-1)
        return anchor_adj + self.diurnal_bias[h] + res * self.anom_sd


def _fuse_highres_v8(
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
    dem_delta_km: torch.Tensor,
    mu_x: torch.Tensor,
    sd_x: torch.Tensor,
    learned_static: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fusiona features GFS a 3 km añadiendo prior adiabático seguro de RH (rh_prior_norm)."""
    B = xt.shape[0]
    grid_b = grid.expand(B, -1, -1, -1)
    f_up = F.grid_sample(f_low, grid_b, mode="bilinear", align_corners=True)
    raw_cat = torch.cat(
        [xt.permute(0, 3, 1, 2), m24_norm.permute(0, 3, 1, 2), s24_norm.permute(0, 3, 1, 2), grad_norm, td_norm],
        dim=1,
    )
    raw_up = F.grid_sample(raw_cat, grid_b, mode="bilinear", align_corners=True)

    topo_b = topo.expand(B, -1, -1, -1)
    u_up, v_up = raw_up[:, 2:3], raw_up[:, 3:4]
    dy_n, dx_n = topo_b[:, 1:2], topo_b[:, 2:3]
    oro_feats = torch.cat([u_up * dx_n + v_up * dy_n, u_up * dy_n - v_up * dx_n], dim=1)

    t_prior = raw_up[:, 1:2] + lapse_delta.expand(B, -1, -1, -1)

    # Prior físico seguro de RH ajustado por elevación mediante Magnus-Tetens
    rh_up_phys = torch.clamp(raw_up[:, 0:1] * sd_x[0] + mu_x[0], 1.0, 100.0)
    t_up_c = torch.clamp((raw_up[:, 1:2] * sd_x[1] + mu_x[1]) - 273.15, -50.0, 50.0)
    a_mag, b_mag = 17.625, 243.04
    gamma = torch.log(rh_up_phys / 100.0) + (a_mag * t_up_c) / (b_mag + t_up_c)
    td_up_c = torch.clamp((b_mag * gamma) / (a_mag - gamma), -60.0, 40.0)
    dkm_b = dem_delta_km[None, None].expand(B, -1, -1, -1)
    t_adj_c = torch.clamp(t_up_c - 6.5 * dkm_b, -50.0, 50.0)
    td_adj_c = torch.clamp(td_up_c - 1.5 * dkm_b, -60.0, 40.0)
    es_t = torch.exp((a_mag * t_adj_c) / (b_mag + t_adj_c))
    es_td = torch.exp((a_mag * td_adj_c) / (b_mag + td_adj_c))
    rh_prior = torch.clamp(100.0 * es_td / (es_t + 1e-6), 1.0, 100.0)
    rh_prior_norm = (rh_prior - mu_x[0]) / sd_x[0]

    static_b = learned_static.expand(B, -1, -1, -1)
    hh_high = h_vec[..., None, None].expand(B, 2, 70, 35)

    # w + 19 + 1 + 1 + 2 + n_topo + 12 + 2 = w + 37 + n_topo canales (n_topo = 10 por defecto -> w + 47)
    fused = torch.cat([f_up, raw_up, t_prior, rh_prior_norm, oro_feats, topo_b, static_b, hh_high], dim=1)
    return fused, topo_b, oro_feats


class MultiScaleBlock2D(nn.Module):
    """Bloque dual-escala paralelo (local 3x3 + mesoescala dilatado 7x7) con SE y FFN."""

    def __init__(self, dim: int, dilation: int = 2, expand: int = 3, dropout: float = 0.18):
        super().__init__()
        pad7 = (7 - 1) * dilation // 2
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw7 = nn.Conv2d(dim, dim, kernel_size=7, padding=pad7, dilation=dilation, groups=dim)
        self.mix = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expand, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pw2 = nn.Conv2d(dim * expand, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-3 * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.mix(torch.cat([self.dw3(x), self.dw7(x)], dim=1))
        return x + self.gamma * self.pw2(self.drop(self.act(self.pw1(self.norm(ms)))))


class DeepWindHead(nn.Module):
    """Cabeza de viento (U, V) con más capacidad (run_cv --wind-head): conv 3x3 de entrada + 2 bloques
    residuales conv 3x3 dilatados (d=2, 4 -> campo receptivo ~±7 celdas ≈ ±21 km) + proyección 1x1."""

    def __init__(self, cin: int, hid: int, dilations=(2, 4)):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(cin, hid, 3, padding=1), nn.GELU())
        self.body = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hid, hid, 3, padding=d, dilation=d), nn.GroupNorm(1, hid), nn.GELU())
            for d in dilations
        ])
        self.out = nn.Conv2d(hid, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inp(x)
        for blk in self.body:
            y = y + blk(y)
        return self.out(y)


def make_wind_head(cin: int, hid: int, wind_head: bool = False) -> nn.Module:
    """Default = la cabeza original (mismos módulos y orden de init -> state_dict/semillas idénticos)."""
    if wind_head:
        return DeepWindHead(cin, hid)
    return nn.Sequential(nn.Conv2d(cin, hid, 3, padding=1), nn.GELU(), nn.Conv2d(hid, 2, 1))


class _DirectResBaseV8(nn.Module):
    """Base común v8: buffers compartidos, TemporalLagMixer, highres fusion y DirectPhysicalAnchor."""

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        dem_delta_km: torch.Tensor,
        mu_x: torch.Tensor,
        sd_x: torch.Tensor,
        anom_sd: torch.Tensor,
        dropout: float = 0.15,
        wind_head: bool = False,
    ):
        super().__init__()
        # n_topo = 10 (train.build_topographic_features) o 10 + extra (topo_extra.py, run_cv --topo-extra):
        # los canales extra van al final, así topo[:, 1:3] (dy, dx) que usa oro_feats no cambia.
        self.n_topo = int(topo_features.shape[0])
        assert self.n_topo >= 3, f"topo_features debe tener >= 3 canales (dem, dy, dx), tiene {self.n_topo}"
        self.wind_head = bool(wind_head)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])
        self.register_buffer("dem_delta_km_2d", dem_delta_km)
        self.temporal_mixer = TemporalLagMixer(d_t=24, num_heads=4, dropout=min(dropout, 0.08))
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)
        self.anchor = DirectPhysicalAnchor(grid, dem_delta_km, mu_x, sd_x, anom_sd)

    def _encode_inputs(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_in, xt, h_vec = _assemble_v7_gfs_tensor(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h.float())
        t_mix = self.temporal_mixer(w_norm)
        gfs_full = torch.cat([gfs_in, t_mix], dim=1)  # 74 + 24 = 98 canales
        return gfs_full, xt, h_vec

    def _fuse(self, f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec):
        return _fuse_highres_v8(
            f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec,
            self.anchor.grid, self.topo, self.lapse_delta, self.dem_delta_km_2d,
            self.anchor.mu_x, self.anchor.sd_x, self.learned_static,
        )


class DirectResConvNeXtV8(_DirectResBaseV8):
    """Familia 1 v8: ConvNeXt 7x7 dilatado con TemporalLagMixer + CrossAttn + DirectPhysicalAnchor."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=84, dropout=0.15,
                 wind_head: bool = False):
        super().__init__(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, dropout, wind_head)
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(98, w, 3, padding=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        self.highres_stem = nn.Sequential(nn.Conv2d(w + 37 + self.n_topo, w, 1), nn.GroupNorm(1, w), nn.GELU())
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w, gfs_ch=w, d_attn=64, num_heads=4)
        self.blocks = nn.Sequential(
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 2, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 7, 3, 3, dropout),
            ConvNeXtBlock2D(w, 7, 1, 3, dropout),
            ConvNeXtBlock2D(w, 5, 1, 2, dropout * 0.5),
        )
        head_in = w + 2 + self.n_topo  # feat + topo (10 [+ extra]) + oro_feats (2)
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = make_wind_head(head_in, w // 2, wind_head)

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_full, xt, h_vec = self._encode_inputs(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        f_low = self.gfs_stem(gfs_full)
        fused, topo_b, oro_feats = self._fuse(f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec)
        feat = self.blocks(self.cross_attn(self.highres_stem(fused), f_low))
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        res = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.anchor(xt, h, res)


class DirectResMultiScaleV8(_DirectResBaseV8):
    """Familia 2 v8: Bloques Multi-Escala (3x3 + 7x7 dilatado) con TemporalLagMixer + DirectPhysicalAnchor."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=96, dropout=0.18,
                 wind_head: bool = False):
        super().__init__(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, dropout, wind_head)
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(98, w, 3, padding=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
            MultiScaleBlock2D(w, dilation=1, expand=2, dropout=dropout * 0.5),
            MultiScaleBlock2D(w, dilation=1, expand=2, dropout=dropout * 0.5),
        )
        self.highres_stem = nn.Sequential(nn.Conv2d(w + 37 + self.n_topo, w, 1), nn.GroupNorm(1, w), nn.GELU())
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w, gfs_ch=w, d_attn=64, num_heads=4)
        self.blocks = nn.Sequential(
            MultiScaleBlock2D(w, dilation=1, expand=3, dropout=dropout),
            MultiScaleBlock2D(w, dilation=2, expand=3, dropout=dropout),
            MultiScaleBlock2D(w, dilation=1, expand=3, dropout=dropout),
            MultiScaleBlock2D(w, dilation=3, expand=3, dropout=dropout),
            MultiScaleBlock2D(w, dilation=1, expand=2, dropout=dropout * 0.5),
        )
        head_in = w + 2 + self.n_topo  # feat + topo (10 [+ extra]) + oro_feats (2)
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = make_wind_head(head_in, w // 2, wind_head)

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_full, xt, h_vec = self._encode_inputs(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        f_low = self.gfs_stem(gfs_full)
        fused, topo_b, oro_feats = self._fuse(f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec)
        feat = self.blocks(self.cross_attn(self.highres_stem(fused), f_low))
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        res = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.anchor(xt, h, res)


class DirectResUNetV8(_DirectResBaseV8):
    """Familia 3 v8: U-Net 3 niveles + CrossAttn en 18x9 convertida a DirectPhysicalAnchor."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=64, dropout=0.14,
                 wind_head: bool = False):
        super().__init__(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, dropout, wind_head)
        self.gfs_enc = nn.Sequential(
            nn.Conv2d(98, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            CompactResBlock(w, dropout=dropout * 0.5),
            CompactResBlock(w, dropout=dropout * 0.5),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(w + 37 + self.n_topo, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(), CompactResBlock(w, dropout)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout * 1.2)
        )
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w * 2, gfs_ch=w, d_attn=64, num_heads=4)
        self.dec2 = nn.Sequential(
            nn.Conv2d(w * 4, w * 2, 3, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), CompactResBlock(w * 2, dropout)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(w * 3, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(), CompactResBlock(w, dropout * 0.5)
        )
        head_in = w + 2 + self.n_topo  # feat + topo (10 [+ extra]) + oro_feats (2)
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w // 2, 3, padding=1), nn.GELU(), nn.Conv2d(w // 2, 2, 1))
        self.head_wind = make_wind_head(head_in, w // 2, wind_head)

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_full, xt, h_vec = self._encode_inputs(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        f_low = self.gfs_enc(gfs_full)
        fused, topo_b, oro_feats = self._fuse(f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec)
        s1 = self.enc1(fused)
        s2 = self.enc2(s1)
        s3 = self.cross_attn(self.enc3(s2), f_low)
        d2 = self.dec2(torch.cat([s2, F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=True)], dim=1))
        d1 = self.dec1(torch.cat([s1, F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=True)], dim=1))
        head_in = torch.cat([d1, topo_b, oro_feats], dim=1)
        res = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.anchor(xt, h, res)


class DirectResFNOV8(_DirectResBaseV8):
    """Familia 4 v8: FNO compacto regularizado (w=32, modes 12x8) con CrossAttn + DirectPhysicalAnchor."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=32, dropout=0.18,
                 wind_head: bool = False):
        super().__init__(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, dropout, wind_head)
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(98, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            FNOBlock2D(w, modes1=5, modes2=4, dropout=dropout * 0.5),
        )
        self.lift = nn.Sequential(nn.Conv2d(w + 37 + self.n_topo, w, 1, bias=False), nn.GroupNorm(8, w), nn.GELU())
        self.cross_attn = GFS2WRFCrossAttention(high_ch=w, gfs_ch=w, d_attn=32, num_heads=4)
        self.fno_layers = nn.Sequential(
            FNOBlock2D(w, 12, 8, dropout),
            FNOBlock2D(w, 12, 8, dropout),
            FNOBlock2D(w, 8, 6, dropout),
        )
        head_in = w + 2 + self.n_topo  # feat + topo (10 [+ extra]) + oro_feats (2)
        self.head_thermo = nn.Sequential(nn.Conv2d(head_in, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 2, 1))
        self.head_wind = make_wind_head(head_in, w, wind_head)

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        gfs_full, xt, h_vec = self._encode_inputs(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        f_low = self.gfs_stem(gfs_full)
        fused, topo_b, oro_feats = self._fuse(f_low, xt, m24_norm, s24_norm, grad_norm, td_norm, h_vec)
        feat = self.fno_layers(self.cross_attn(self.lift(fused), f_low))
        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        res = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1).permute(0, 2, 3, 1)
        return self.anchor(xt, h, res)


class EnsembleV8(nn.Module):
    """Módulo único v8 que combina las 4 familias DirectRes en espacio físico (B, 70, 35, 4)."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, blend_weights: np.ndarray):
        super().__init__()
        self.m_conv = DirectResConvNeXtV8(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=84, dropout=0.0)
        self.m_ms = DirectResMultiScaleV8(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=96, dropout=0.0)
        self.m_unet = DirectResUNetV8(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=64, dropout=0.0)
        self.m_fno = DirectResFNOV8(grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=32, dropout=0.0)
        self.register_buffer("blend_weights", torch.tensor(blend_weights, dtype=torch.float32))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        y0 = self.m_conv(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y1 = self.m_ms(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y2 = self.m_unet(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        y3 = self.m_fno(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
        bw = self.blend_weights
        return (
            bw[0].view(1, 1, 1, 4) * y0
            + bw[1].view(1, 1, 1, 4) * y1
            + bw[2].view(1, 1, 1, 4) * y2
            + bw[3].view(1, 1, 1, 4) * y3
        )


def optimize_blend_weights_simplex(
    oofs: List[np.ndarray], y_val: np.ndarray, clim_rmse_val: np.ndarray, step: float = 0.05, reg_coef: float = 0.0015
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimiza pesos por variable en el simplex con pasos `step` y regularización suave hacia uniforme."""
    n_steps = int(round(1.0 / step))
    combos = []
    for i0 in range(n_steps + 1):
        for i1 in range(n_steps + 1 - i0):
            for i2 in range(n_steps + 1 - i0 - i1):
                i3 = n_steps - i0 - i1 - i2
                combos.append((i0 * step, i1 * step, i2 * step, i3 * step))
    W_grid = np.asarray(combos, dtype=np.float32)  # (C, 4)
    reg_vec = reg_coef * ((W_grid - 0.25) ** 2).sum(axis=1)  # (C,)

    blend_w = np.zeros((4, 4), dtype=np.float32)
    oof_opt = np.empty_like(oofs[0])
    for v_idx in range(4):
        M = np.stack([o[..., v_idx].reshape(-1) for o in oofs], axis=0)  # (4, P)
        y_flat = y_val[..., v_idx].reshape(-1)
        # Matriz de covarianza de errores: mse(w) = w^T E w donde E_{jk} = mean(e_j * e_k)
        E = M - y_flat[None, :]  # (4, P)
        C_err = (E @ E.T) / float(E.shape[1])  # (4, 4)
        mses = np.einsum("ci,ij,cj->c", W_grid, C_err, W_grid)
        rmses = np.sqrt(np.maximum(mses, 1e-12))
        objs = rmses / float(clim_rmse_val[v_idx]) + reg_vec
        best_idx = int(np.argmin(objs))
        best_wv = W_grid[best_idx]
        blend_w[:, v_idx] = best_wv
        oof_opt[..., v_idx] = sum(float(best_wv[k]) * oofs[k][..., v_idx] for k in range(4))
    return blend_w, oof_opt


def main(epochs: int = 24, run_multiblock: bool = False) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train v8] Dispositivo: {device.upper()} | Épocas Holdout: {epochs}")

    tr = np.load(DATA_DIR / "train.npz")
    st = dict(np.load(DATA_DIR / "static.npz"))
    Xtr, Ytr, ttr = tr["X"], tr["Y"], tr["times"].astype(str)
    dem = st["dem"].astype(np.float32)
    dem_delta_km_np = ((dem - smooth_2d(dem, k=9)) / 1000.0).astype(np.float32)
    dem_delta_km_t = torch.tensor(dem_delta_km_np, dtype=torch.float32, device=device)

    val_mask = ttr >= "2024061600"
    trn_mask = ~val_mask

    d = prepare_split_tensors(Xtr, Ytr, ttr, st, trn_mask, val_mask, device)
    y_val = Ytr[val_mask]
    clim_rmse_val = d["clim_rmse_val"]

    specs = [
        (
            "DirectResConvNeXtV8",
            DirectResConvNeXtV8(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], dem_delta_km_t, d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=84, dropout=0.15),
            1.8e-3, 1.5e-2, 0.045, 42,
        ),
        (
            "DirectResMultiScaleV8",
            DirectResMultiScaleV8(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], dem_delta_km_t, d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=96, dropout=0.18),
            1.6e-3, 1.8e-2, 0.050, 102,
        ),
        (
            "DirectResUNetV8",
            DirectResUNetV8(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], dem_delta_km_t, d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=64, dropout=0.14),
            1.5e-3, 1.4e-2, 0.040, 203,
        ),
        (
            "DirectResFNOV8",
            DirectResFNOV8(d["grid_t"], d["topo_t"], d["lapse_delta_norm"], dem_delta_km_t, d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=32, dropout=0.18),
            1.8e-3, 2.5e-2, 0.050, 304,
        ),
    ]

    oofs: List[np.ndarray] = []
    member_results = []
    lapse_logs = {}
    for name, model, lr, wd, n_std, seed in specs:
        best_state, oof, sc, best_ep, skill_d, rmse_d = train_member_v7(
            name, model, trn_mask, val_mask,
            d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
            d["sample_weights"], d["clim_rmse_trn_t"], clim_rmse_val, device,
            epochs=epochs, lr=lr, weight_decay=wd, noise_std=n_std, seed=seed,
        )
        assert not np.isnan(oof).any(), f"NaN detectado en {name}"
        lr_h = best_state["anchor.lapse_rate_h"].squeeze().numpy()
        rh_h = best_state["anchor.rh_elev_h"].squeeze().numpy()
        lapse_logs[name] = (lr_h, rh_h)
        oofs.append(oof)
        member_results.append((name, sc, best_ep, skill_d, rmse_d))
        del best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    oof_eq = sum(oofs) / len(oofs)
    sc_eq, rmse_eq, skill_eq = evaluate_skill_score(oof_eq, y_val, clim_rmse_val)

    blend_w, oof_opt = optimize_blend_weights_simplex(oofs, y_val, clim_rmse_val, step=0.05, reg_coef=0.0015)
    sc_opt, rmse_opt, skill_opt = evaluate_skill_score(oof_opt, y_val, clim_rmse_val)

    print("\n" + "=" * 88)
    print("RESUMEN FASE A — HOLDOUT TEMPORAL (>= 2024061600) — SUITE v8 (FULL DIRECTRES)")
    print("=" * 88)
    for name, sc, ep, sk, rm in member_results:
        print(f"  {name:24s} | Ep={ep:02d} | Skill={sc:.4f} | Var={sk}")
    print(f"  {'Ensamble Uniforme 25%':24s} |          | Skill={sc_eq:.4f} | Var={skill_eq}")
    print(f"  {'Ensamble Óptimo Reg.':24s} |          | Skill={sc_opt:.4f} | Var={skill_opt}")
    print(f"  RMSE Ensamble Óptimo     : {rmse_opt}")
    print(f"  Pesos (Conv, MS, UNet, FNO) por [RH, T, U, V]:\n{blend_w}")
    lr0, rh0 = lapse_logs["DirectResConvNeXtV8"]
    print(f"  Lapse-Rate T aprendida (K/km) 00UTC={lr0[0]:.2f}, 12UTC={lr0[12]:.2f}, 18UTC={lr0[18]:.2f}")
    print(f"  Lapse-Rate RH aprendida (%/km) 00UTC={rh0[0]:.2f}, 12UTC={rh0[12]:.2f}, 18UTC={rh0[18]:.2f}")
    print("=" * 88)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    logger.log_experiment(
        model_name="ensemble_v8_holdout",
        metric_name="skill_score",
        fold_scores=[sc_opt],
        params={
            "members": [{"name": n, "skill": round(s, 4), "best_ep": ep} for n, s, ep, _, _ in member_results],
            "blend_weights": blend_w.tolist(),
            "uniform_skill": round(sc_eq, 4),
            "val_rmse_per_var": rmse_opt,
            "val_skill_per_var": skill_opt,
        },
        features=["V7_LAGS_+-24h", "TemporalLagMixer", "DirectPhysicalAnchor_HourlyLapse", "RH_Prior_Magnus", "GFS2WRF_CrossAttn"],
        oof_preds=oof_opt,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=42,
        notes=f"Skill={sc_opt:.4f} (eq={sc_eq:.4f}) | {skill_opt}",
    )

    if run_multiblock:
        print("[Aviso] Modo multiblock omitido en holdout rápido.")

    return sc_opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación v8 — Hackathon IA PUCP 2026")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--no-multiblock", action="store_true")
    args = parser.parse_args()
    main(epochs=args.epochs, run_multiblock=not args.no_multiblock)
