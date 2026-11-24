"""
2da Arquitectura Diversa — PhysicsConvNeXt (Large-Kernel 7x7 + Multi-Scale Dilated Context).
Hackathon IA PUCP 2026 · AIRE-ANDINO.

Por qué es arquitectónicamente distinta y complementaria a PhysicsUNet v3:
- PhysicsUNet usa convoluciones 3x3 con reducción/ampliación de resolución (70x35 -> 35x18 -> 18x9).
- PhysicsConvNeXt mantiene la resolución completa de 3 km (70x35) sin comprimir los picos del DEM,
  usando bloques modernos ConvNeXt con kernels grandes 7x7 depthwise (ven 21 km x 21 km de golpe)
  y convoluciones dilatadas (dilation=2 -> 39 km x 39 km) para capturar el encajonamiento del viento
  a lo largo de todo el Callejón de Huaylas.
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


class ConvNeXtBlock2D(nn.Module):
    """Bloque ConvNeXt moderno con kernel grande (7x7 depthwise), inverted bottleneck y LayerScale."""

    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1, expand: int = 3, dropout: float = 0.12):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # Equivalente a LayerNorm por canal en 2D
        self.pw1 = nn.Conv2d(dim, dim * expand, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pw2 = nn.Conv2d(dim * expand, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-3 * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        return residual + self.gamma * x


class PhysicsConvNeXt(nn.Module):
    """
    2da Arquitectura: Backbone isotrópico a resolución completa 70x35 con kernels grandes 7x7
    y contexto multi-escala dilatado, optimizado para viento en valles y frentes de humedad.
    """

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        w: int = 80,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("grid", grid)
        self.register_buffer("topo", topo_features[None])
        self.register_buffer("lapse_delta", lapse_delta_norm[None, None])

        # Codificador en baja resolución GFS (17x13) con ConvNeXt 5x5
        in_gfs_ch = 50
        self.gfs_stem = nn.Sequential(
            nn.Conv2d(in_gfs_ch, w, kernel_size=3, padding=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
            ConvNeXtBlock2D(w, kernel_size=5, dilation=1, expand=2, dropout=dropout * 0.5),
            ConvNeXtBlock2D(w, kernel_size=5, dilation=1, expand=2, dropout=dropout * 0.5),
        )

        # Embedding estático espacial
        self.learned_static = nn.Parameter(torch.zeros(1, 12, 70, 35))
        nn.init.normal_(self.learned_static, mean=0.0, std=0.01)

        # Fusión en alta resolución 70x35:
        # w + 9 (xt, m24, td) + 1 (t_prior) + 2 (oro_feats) + 10 (topo) + 12 (static) + 2 (hour)
        fused_ch = w + 9 + 1 + 2 + 10 + 12 + 2
        self.highres_stem = nn.Sequential(
            nn.Conv2d(fused_ch, w, kernel_size=1),
            nn.GroupNorm(1, w),
            nn.GELU(),
        )

        # Torre ConvNeXt a resolución completa 70x35 (alternando campo receptivo local y regional)
        self.blocks = nn.Sequential(
            ConvNeXtBlock2D(w, kernel_size=7, dilation=1, expand=3, dropout=dropout),  # 21 km
            ConvNeXtBlock2D(w, kernel_size=7, dilation=2, expand=3, dropout=dropout),  # 39 km
            ConvNeXtBlock2D(w, kernel_size=7, dilation=1, expand=3, dropout=dropout),  # 21 km
            ConvNeXtBlock2D(w, kernel_size=7, dilation=3, expand=3, dropout=dropout),  # 57 km (toda la cordillera)
            ConvNeXtBlock2D(w, kernel_size=7, dilation=1, expand=3, dropout=dropout),  # refinamiento local
            ConvNeXtBlock2D(w, kernel_size=5, dilation=1, expand=2, dropout=dropout * 0.5),
        )

        # Cabezales especializados separados para RH, T, U, V
        head_in = w + 10 + 2
        self.head_thermo = nn.Sequential(
            nn.Conv2d(head_in, w // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, kernel_size=1),
        )
        self.head_wind = nn.Sequential(
            nn.Conv2d(head_in, w // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(w // 2, 2, kernel_size=1),
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
        feat = self.blocks(self.highres_stem(fused))

        head_in = torch.cat([feat, topo_b, oro_feats], dim=1)
        out = torch.cat([self.head_thermo(head_in), self.head_wind(head_in)], dim=1)
        return out.permute(0, 2, 3, 1)


def train_convnext(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 2.0e-3,
    width: int = 80,
    dropout: float = 0.15,
    noise_std: float = 0.05,
    seed: int = 102,
) -> float:
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train ConvNeXt] Dispositivo: {device.upper()} | Width: {width} | Seed: {seed}")

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

    model = PhysicsConvNeXt(
        grid=grid_t,
        topo_features=topo_t,
        lapse_delta_norm=lapse_delta_t,
        w=width,
        dropout=dropout,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Modelo] PhysicsConvNeXt parámetros entrenables: {n_params:,}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-2)
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
    print(f"🏆 MEJOR SKILL SCORE VALIDACIÓN (PhysicsConvNeXt · {best_source}): {best_score:.4f}")
    print(f"   Skill por variable : {best_skill_dict}")
    print(f"   RMSE por variable  : {best_rmse_dict}")
    print("=" * 80)

    model.load_state_dict(best_state)
    val_oof = predict_array(model, Wtr_n[val_mask], M24_tr_n[val_mask], Td_tr_n[val_mask], h_tr[val_mask])
    test_preds = predict_array(model, Wte_n, M24_te_n, Td_te_n, h_te)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    exp_id = logger.log_experiment(
        model_name="physics_convnext_v4",
        metric_name="skill_score",
        fold_scores=[best_score],
        params={
            "arch": "physics_convnext_v4",
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
            "LargeKernel_7x7_Dilated_ConvNeXt",
        ],
        oof_preds=val_oof,
        test_preds=test_preds,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=seed,
        notes=f"Skill={best_score:.4f} ({best_source}) | Skills={best_skill_dict}",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    convnext_path = MODELS_DIR / "physics_convnext_v4.pt"
    torch.save(best_state, convnext_path)
    print(f"[Export] Pesos ConvNeXt guardados en {convnext_path} ({os.path.getsize(convnext_path)/1e6:.2f} MB)")

    # Evaluar inmediatamente el ENSAMBLE (PhysicsUNet v3 + PhysicsConvNeXt v4) en validación temporal
    unet_oofs = sorted(EXPERIMENTS_DIR.glob("*physics_unet_v3*/oof_preds.npy"))
    if unet_oofs:
        oof_unet = np.load(unet_oofs[-1])
        print("\n=== EVALUACIÓN DE ENSAMBLE EN VALIDACIÓN (PhysicsUNet v3 + PhysicsConvNeXt v4) ===")
        best_blend_sc, best_w = -1e9, 0.5
        for w_c in np.linspace(0.0, 1.0, 21):
            blend_oof = (1.0 - w_c) * oof_unet + w_c * val_oof
            sc_b, _, _ = evaluate_skill_score(blend_oof, Ytr[val_mask], clim_rmse_val)
            if sc_b > best_blend_sc:
                best_blend_sc, best_w = sc_b, float(w_c)

        # Pesos óptimos por variable
        opt_weights_var = []
        blend_per_var = np.empty_like(val_oof)
        for v_idx in range(4):
            best_v_rmse, best_vw = 1e9, 0.5
            for w_c in np.linspace(0.0, 1.0, 41):
                pred_v = (1.0 - w_c) * oof_unet[..., v_idx] + w_c * val_oof[..., v_idx]
                rmse_v = float(np.sqrt(((pred_v - Ytr[val_mask, ..., v_idx]) ** 2).mean()))
                if rmse_v < best_v_rmse:
                    best_v_rmse, best_vw = rmse_v, float(w_c)
            opt_weights_var.append(round(best_vw, 3))
            blend_per_var[..., v_idx] = (1.0 - best_vw) * oof_unet[..., v_idx] + best_vw * val_oof[..., v_idx]

        sc_var_blend, rmse_vb, skill_vb = evaluate_skill_score(blend_per_var, Ytr[val_mask], clim_rmse_val)
        print(f"  Solo PhysicsUNet v3       : {evaluate_skill_score(oof_unet, Ytr[val_mask], clim_rmse_val)[0]:.4f}")
        print(f"  Solo PhysicsConvNeXt v4   : {best_score:.4f}")
        print(f"  Ensamble peso global      : {best_blend_sc:.4f} (w_convnext={best_w:.2f})")
        print(f"  Ensamble óptimo/variable  : {sc_var_blend:.4f} (w_convnext={dict(zip(VARS, opt_weights_var))})")
        print(f"    -> Skill/var            : {skill_vb}")
        print(f"    -> RMSE/var             : {rmse_vb}")

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PhysicsConvNeXt — Hackathon IA PUCP 2026")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=102)
    args = parser.parse_args()

    train_convnext(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        dropout=args.dropout,
        noise_std=args.noise_std,
        seed=args.seed,
    )
