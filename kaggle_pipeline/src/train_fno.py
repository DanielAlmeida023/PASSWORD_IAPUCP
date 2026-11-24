"""
3ra Arquitectura Diversa — PhysicsFNO (Fourier Neural Operator 2D + Rama Topográfica Local).
Hackathon IA PUCP 2026 · AIRE-ANDINO.

Por qué es la 3ra y última familia del ensamble:
- PhysicsUNet v3     : Convoluciones 3x3 con pirámide multi-escala (fuerte en viento local).
- PhysicsConvNeXt v4 : Kernels grandes 7x7 dilatados sin compresión (fuerte en temperatura y relieve).
- PhysicsFNO v5      : Operador Neuronal de Fourier 2D (rfft2 -> mezcla espectral global -> irfft2),
                       misma base matemática que FourCastNet de NVIDIA. Captura ondas atmosféricas y
                       campos globales de humedad/viento sobre toda la Cordillera Blanca de golpe.
"""

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from seed import seed_everything
from experiment_logger import ExperimentLogger
from train import (
    VARS,
    extract_hour_utc,
    compute_climatology,
    smooth_2d,
    build_topographic_features,
    build_sampling_grid,
    evaluate_skill_score,
)
from train_v3 import (
    CENTERED_LAGS,
    compute_dewpoint_c,
    build_centered_windows_and_daily_mean,
    update_ema,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


class SpectralConv2d(nn.Module):
    """Convolución espectral 2D en el dominio de frecuencias (Fourier Neural Operator)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 16, modes2: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Almacenado como tensores reales (real, imag) para máxima compatibilidad con AdamW y torch.save
        self.w1_r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w1_i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w2_r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w2_i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    @staticmethod
    def _compl_mul2d(xr: torch.Tensor, xi: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor):
        # (B, in, m1, m2) x (in, out, m1, m2) -> (B, out, m1, m2)
        or_ = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
        oi_ = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
        return torch.complex(or_, oi_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2 + 1)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        # Frecuencias bajas positivas
        x1 = x_ft[:, :, :m1, :m2]
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x1.real, x1.imag, self.w1_r[:, :, :m1, :m2], self.w1_i[:, :, :m1, :m2]
        )
        # Frecuencias bajas negativas (eje vertical)
        x2 = x_ft[:, :, -m1:, :m2]
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x2.real, x2.imag, self.w2_r[:, :, :m1, :m2], self.w2_i[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock2D(nn.Module):
    """Bloque híbrido: Operador espectral global (Fourier) + Convolución local (relieve DEM)."""

    def __init__(self, dim: int, modes1: int = 16, modes2: int = 10, dropout: float = 0.12):
        super().__init__()
        self.spectral = SpectralConv2d(dim, dim, modes1=modes1, modes2=modes2)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral(x) + self.pw(self.local_conv(x))
        h = self.drop(self.act(self.norm(h)))
        return x + h


class PhysicsFNO(nn.Module):
    """
    3ra Arquitectura: Fourier Neural Operator 2D guiado por física andina.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        w: int = 48,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])

        # Codificador espectral + local en baja resolución GFS (17x13)
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(50, w, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
            FNOBlock2D(w, modes1=6, modes2=5, dropout=dropout * 0.5),
        )

        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        fused_ch = w + 9 + 1 + 2 + 10 + 12 + 2
        self.lift = nn.Sequential(
            nn.Conv2d(fused_ch, w, kernel_size=1, bias=False),
            nn.GroupNorm(8, w),
            nn.GELU(),
        )

        # 4 capas FNO en resolución completa 70x35 (capturan ondas regionales + detalle topográfico)
        self.fno_layers = nn.Sequential(
            FNOBlock2D(w, modes1=16, modes2=10, dropout=dropout),
            FNOBlock2D(w, modes1=16, modes2=10, dropout=dropout),
            FNOBlock2D(w, modes1=12, modes2=8, dropout=dropout),
            FNOBlock2D(w, modes1=12, modes2=8, dropout=dropout * 0.5),
        )

        head_in = w + 10 + 2
        self.head_thermo = nn.Sequential(
            nn.Conv2d(head_in, w, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, 2, kernel_size=1),
        )
        self.head_wind = nn.Sequential(
            nn.Conv2d(head_in, w, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, 2, kernel_size=1),
        )

    def forward(
        self,
        w_norm: torch.Tensor,
        m24_norm: torch.Tensor,
        td_norm: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        B = w_norm.shape[0]
        xt = w_norm[:, 3]
        xm3 = w_norm[:, 1]
        xp3 = w_norm[:, 5]

        wspd_list = []
        for idx in (1, 3, 5):
            u_i, v_i = w_norm[:, idx, :, :, 2], w_norm[:, idx, :, :, 3]
            wspd_list.append(torch.sqrt(u_i**2 + v_i**2 + 1e-6))
        wspd = torch.stack(wspd_list, dim=1)

        d_prev = (xt - xm3).permute(0, 3, 1, 2)
        d_next = (xp3 - xt).permute(0, 3, 1, 2)
        d_diurnal = (xt - m24_norm).permute(0, 3, 1, 2)

        w_flat = w_norm.permute(0, 1, 4, 2, 3).reshape(B, 28, 17, 13)
        m24_flat = m24_norm.permute(0, 3, 1, 2)

        ang = 2.0 * np.pi * h / 24.0
        h_vec = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
        hh_low = h_vec[..., None, None].expand(B, 2, 17, 13)

        gfs_in = torch.cat([w_flat, m24_flat, td_norm, wspd, d_prev, d_next, d_diurnal, hh_low], dim=1)
        f_low = self.gfs_stem(gfs_in)

        grid_b = self.grid.expand(B, -1, -1, -1)
        f_up = F.grid_sample(f_low, grid_b, mode="bilinear", align_corners=True)
        raw_cat = torch.cat([xt.permute(0, 3, 1, 2), m24_flat, td_norm], dim=1)
        raw_up = F.grid_sample(raw_cat, grid_b, mode="bilinear", align_corners=True)

        topo_b = self.topo.expand(B, -1, -1, -1)
        u_up, v_up = raw_up[:, 2:3], raw_up[:, 3:4]
        dy_n, dx_n = topo_b[:, 1:2], topo_b[:, 2:3]
        oro_feats = torch.cat([u_up * dx_n + v_up * dy_n, u_up * dy_n - v_up * dx_n], dim=1)

        t_prior = raw_up[:, 1:2] + self.lapse_delta.expand(B, -1, -1, -1)
        static_b = self.learned_static.expand(B, -1, -1, -1)
        hh_high = h_vec[..., None, None].expand(B, 2, 70, 35)

        fused = torch.cat([f_up, raw_up, t_prior, oro_feats, topo_b, static_b, hh_high], dim=1)
        feat = self.fno_layers(self.lift(fused))

        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        out = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1)
        return out.permute(0, 2, 3, 1)


def train_fno(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 2.0e-3,
    width: int = 48,
    dropout: float = 0.12,
    noise_std: float = 0.05,
    seed: int = 203,
) -> float:
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train FNO] Dispositivo: {device.upper()} | Width: {width} | Seed: {seed}")

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
    clim_rmse_trn = np.sqrt(((clim_trn[h_tr[trn_mask]] - Ytr[trn_mask]) ** 2).mean(axis=(0, 1, 2))).astype(np.float32)

    Wtr, M24_tr = build_centered_windows_and_daily_mean(Xtr, ttr, CENTERED_LAGS)
    Wte, M24_te = build_centered_windows_and_daily_mean(Xte, tte, CENTERED_LAGS)

    Td_tr = compute_dewpoint_c(Xtr[..., 1], Xtr[..., 0])[:, None, :, :]
    Td_te = compute_dewpoint_c(Xte[..., 1], Xte[..., 0])[:, None, :, :]

    mu_x = Xtr[trn_mask].mean(axis=(0, 1, 2)).astype(np.float32)
    sd_x = Xtr[trn_mask].std(axis=(0, 1, 2)).astype(np.float32)
    mu_td = float(Td_tr[trn_mask].mean())
    sd_td = float(Td_tr[trn_mask].std() + 1e-6)
    anom_sd = (Ytr[trn_mask] - clim_trn[h_tr[trn_mask]]).std(axis=(0, 1, 2)).astype(np.float32)

    months_tr = pd.to_datetime(ttr, format="%Y%m%d%H").month.values
    month_w_map = {2: 0.55, 3: 0.80, 4: 0.90, 5: 1.15, 6: 1.40}
    sample_weights = np.array([month_w_map.get(int(m), 1.0) for m in months_tr], dtype=np.float32)
    sample_weights[trn_mask] /= sample_weights[trn_mask].mean()

    grid_t = build_sampling_grid(st, device)
    topo_np = build_topographic_features(dem, st["wrf_lats"], st["wrf_lons"])
    topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)

    dem_smooth = smooth_2d(dem, k=9)
    lapse_delta = (-0.0065 * (dem - dem_smooth)) / float(sd_x[1])
    lapse_delta_t = torch.tensor(lapse_delta, dtype=torch.float32, device=device)

    model = PhysicsFNO(
        grid=grid_t,
        topo_features=topo_t,
        lapse_delta_norm=lapse_delta_t,
        w=width,
        dropout=dropout,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] PhysicsFNO parámetros entrenables: {n_params:,}")

    CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
    ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)
    CLIM_RMSE_TRN_T = torch.tensor(clim_rmse_trn, dtype=torch.float32, device=device)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
    itr = np.where(trn_mask)[0]
    steps_per_epoch = int(np.ceil(len(itr) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15
    )

    best_score = -1e9
    best_state = None
    best_rmse_dict: Dict[str, float] = {}
    best_skill_dict: Dict[str, float] = {}
    best_source = "LIVE"

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

            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
                mb = mb + torch.randn_like(mb) * (noise_std * 0.5)

            pred_anom = model(xb, mb, tb, hb.float())
            y_pred = CLIM_T[hb] + pred_anom * ASD_T

            err_sq = (y_pred - yb_true) ** 2
            per_sample_mse = err_sq.mean(dim=(1, 2))
            weighted_mse = (per_sample_mse * wb[:, None]).sum(dim=0) / (wb.sum() + 1e-8)
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
    print(f"🏆 MEJOR SKILL SCORE VALIDACIÓN (PhysicsFNO v5 · {best_source}): {best_score:.4f}")
    print(f"   Skill por variable : {best_skill_dict}")
    print(f"   RMSE por variable  : {best_rmse_dict}")
    print("=" * 80)

    model.load_state_dict(best_state)
    val_oof = predict_array(model, Wtr_n[val_mask], M24_tr_n[val_mask], Td_tr_n[val_mask], h_tr[val_mask])
    test_preds = predict_array(model, Wte_n, M24_te_n, Td_te_n, h_te)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    logger.log_experiment(
        model_name="physics_fno_v5",
        metric_name="skill_score",
        fold_scores=[best_score],
        params={
            "arch": "physics_fno_v5",
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
            "Fourier_Neural_Operator_2D",
        ],
        oof_preds=val_oof,
        test_preds=test_preds,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=seed,
        notes=f"Skill={best_score:.4f} ({best_source}) | Skills={best_skill_dict}",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fno_path = MODELS_DIR / "physics_fno_v5.pt"
    torch.save(best_state, fno_path)
    print(f"[Export] Pesos FNO guardados en {fno_path} ({os.path.getsize(fno_path)/1e6:.2f} MB)")

    # Evaluar ENSAMBLE DE 3 FAMILIAS (PhysicsUNet v3 + PhysicsConvNeXt v4 + PhysicsFNO v5)
    unet_oofs = sorted(EXPERIMENTS_DIR.glob("*physics_unet_v3*/oof_preds.npy"))
    conv_oofs = sorted(EXPERIMENTS_DIR.glob("*physics_convnext_v4*/oof_preds.npy"))
    if unet_oofs and conv_oofs:
        oof_u = np.load(unet_oofs[-1])
        oof_c = np.load(conv_oofs[-1])
        oof_f = val_oof
        y_val = Ytr[val_mask]

        print("\n=== EVALUACIÓN ENSAMBLE DE 3 FAMILIAS (UNet + ConvNeXt + FNO) ===")
        blend3 = np.empty_like(oof_f)
        best_weights = {}
        grid_vals = np.linspace(0.0, 1.0, 21)
        for v_idx, v_name in enumerate(VARS):
            best_rmse, best_w = 1e9, (1 / 3, 1 / 3, 1 / 3)
            for wu in grid_vals:
                for wc in grid_vals:
                    wf = 1.0 - wu - wc
                    if wf < -1e-6:
                        continue
                    wf = max(0.0, wf)
                    pred_v = wu * oof_u[..., v_idx] + wc * oof_c[..., v_idx] + wf * oof_f[..., v_idx]
                    rmse_v = float(np.sqrt(((pred_v - y_val[..., v_idx]) ** 2).mean()))
                    if rmse_v < best_rmse:
                        best_rmse = rmse_v
                        best_w = (round(float(wu), 2), round(float(wc), 2), round(float(wf), 2))
            best_weights[v_name] = best_w
            blend3[..., v_idx] = (
                best_w[0] * oof_u[..., v_idx]
                + best_w[1] * oof_c[..., v_idx]
                + best_w[2] * oof_f[..., v_idx]
            )

        sc3, rmse3, skill3 = evaluate_skill_score(blend3, y_val, clim_rmse_val)
        sc_eq, _, _ = evaluate_skill_score((oof_u + oof_c + oof_f) / 3.0, y_val, clim_rmse_val)
        print(f"  1. Solo PhysicsUNet v3       : {evaluate_skill_score(oof_u, y_val, clim_rmse_val)[0]:.4f}")
        print(f"  2. Solo PhysicsConvNeXt v4   : {evaluate_skill_score(oof_c, y_val, clim_rmse_val)[0]:.4f}")
        print(f"  3. Solo PhysicsFNO v5        : {best_score:.4f}")
        print(f"  Ensamble 3 familias (1/3 c/u): {sc_eq:.4f}")
        print(f"  Ensamble 3 familias (óptimo) : {sc3:.4f}")
        print(f"    -> Pesos (UNet, ConvNeXt, FNO): {best_weights}")
        print(f"    -> Skill/var               : {skill3}")
        print(f"    -> RMSE/var                : {rmse3}")

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PhysicsFNO v5 — Hackathon IA PUCP 2026")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=203)
    args = parser.parse_args()

    train_fno(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        dropout=args.dropout,
        noise_std=args.noise_std,
        seed=args.seed,
    )
