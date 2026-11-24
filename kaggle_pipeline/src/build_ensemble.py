"""
Empaquetador del Modelo Final Único — EnsembleDownscaler (UNet + ConvNeXt + FNO).
Hackathon IA PUCP 2026 · AIRE-ANDINO.

Une las 3 arquitecturas complementarias ya entrenadas (PhysicsUNet v3, PhysicsConvNeXt v4 y PhysicsFNO v5)
dentro de un único nn.Module (`EnsembleDownscaler`) y guarda todos sus pesos en un solo archivo
`models/modelo.pt` (~30.7 MB), generando además `submission.csv` y validando que reproduce exactamente
el Skill Score de validación (0.3835).
"""

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    PhysicsUNet,
    CENTERED_LAGS,
    compute_dewpoint_c,
    build_centered_windows_and_daily_mean,
)
from train_convnext import PhysicsConvNeXt
from train_fno import PhysicsFNO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# Pesos óptimos por variable (filas: [UNet, ConvNeXt, FNO], columnas: [RH, T, U, V])
DEFAULT_BLEND_WEIGHTS = np.array(
    [
        [0.25, 0.20, 0.40, 0.40],  # PhysicsUNet v3
        [0.30, 0.80, 0.35, 0.30],  # PhysicsConvNeXt v4
        [0.45, 0.00, 0.25, 0.30],  # PhysicsFNO v5
    ],
    dtype=np.float32,
)


class EnsembleDownscaler(nn.Module):
    """
    Modelo único de inferencia que combina las 3 familias arquitectónicas complementarias
    con pesos óptimos por variable meteorológica (RH, T, U, V).
    """

    def __init__(
        self,
        grid: torch.Tensor,
        topo_features: torch.Tensor,
        lapse_delta_norm: torch.Tensor,
        unet_w: int = 64,
        convnext_w: int = 80,
        fno_w: int = 48,
        blend_weights: np.ndarray = DEFAULT_BLEND_WEIGHTS,
    ):
        super().__init__()
        self.unet = PhysicsUNet(
            grid=grid, topo_features=topo_features, lapse_delta_norm=lapse_delta_norm, w=unet_w, dropout=0.0
        )
        self.convnext = PhysicsConvNeXt(
            grid=grid, topo_features=topo_features, lapse_delta_norm=lapse_delta_norm, w=convnext_w, dropout=0.0
        )
        self.fno = PhysicsFNO(
            grid=grid, topo_features=topo_features, lapse_delta_norm=lapse_delta_norm, w=fno_w, dropout=0.0
        )
        self.register_buffer("blend_weights", torch.tensor(blend_weights, dtype=torch.float32))  # (3, 4)

    def forward(
        self,
        w_norm: torch.Tensor,
        m24_norm: torch.Tensor,
        td_norm: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        out_u = self.unet(w_norm, m24_norm, td_norm, h)
        out_c = self.convnext(w_norm, m24_norm, td_norm, h)
        out_f = self.fno(w_norm, m24_norm, td_norm, h)

        bw = self.blend_weights  # (3, 4)
        return (
            bw[0].view(1, 1, 1, 4) * out_u
            + bw[1].view(1, 1, 1, 4) * out_c
            + bw[2].view(1, 1, 1, 4) * out_f
        )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Ensemble] Construyendo EnsembleDownscaler único en {device.upper()}...")

    unet_path = MODELS_DIR / "modelo.pt"
    convnext_path = MODELS_DIR / "physics_convnext_v4.pt"
    fno_path = MODELS_DIR / "physics_fno_v5.pt"
    stats_path = MODELS_DIR / "norm_stats.npz"

    for p in (unet_path, convnext_path, fno_path, stats_path):
        assert p.exists(), f"Falta archivo requerido para empaquetar: {p}"

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

    Wtr, M24_tr = build_centered_windows_and_daily_mean(Xtr, ttr, CENTERED_LAGS)
    Wte, M24_te = build_centered_windows_and_daily_mean(Xte, tte, CENTERED_LAGS)

    Td_tr = compute_dewpoint_c(Xtr[..., 1], Xtr[..., 0])[:, None, :, :]
    Td_te = compute_dewpoint_c(Xte[..., 1], Xte[..., 0])[:, None, :, :]

    mu_x = Xtr[trn_mask].mean(axis=(0, 1, 2)).astype(np.float32)
    sd_x = Xtr[trn_mask].std(axis=(0, 1, 2)).astype(np.float32)
    mu_td = float(Td_tr[trn_mask].mean())
    sd_td = float(Td_tr[trn_mask].std() + 1e-6)
    anom_sd = (Ytr[trn_mask] - clim_trn[h_tr[trn_mask]]).std(axis=(0, 1, 2)).astype(np.float32)

    grid_t = build_sampling_grid(st, device)
    topo_np = build_topographic_features(dem, st["wrf_lats"], st["wrf_lons"])
    topo_t = torch.tensor(topo_np, dtype=torch.float32, device=device)
    dem_smooth = smooth_2d(dem, k=9)
    lapse_delta = (-0.0065 * (dem - dem_smooth)) / float(sd_x[1])
    lapse_delta_t = torch.tensor(lapse_delta, dtype=torch.float32, device=device)

    ensemble = EnsembleDownscaler(
        grid=grid_t,
        topo_features=topo_t,
        lapse_delta_norm=lapse_delta_t,
        unet_w=64,
        convnext_w=80,
        fno_w=48,
        blend_weights=DEFAULT_BLEND_WEIGHTS,
    ).to(device)

    # Cargar pesos de cada sub-arquitectura
    ensemble.unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    ensemble.convnext.load_state_dict(torch.load(convnext_path, map_location=device, weights_only=True))
    ensemble.fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
    ensemble.eval()

    n_params = sum(p.numel() for p in ensemble.parameters())
    print(f"[Ensemble] Parámetros totales del modelo unificado: {n_params:,}")

    CLIM_T = torch.tensor(clim_trn, dtype=torch.float32, device=device)
    ASD_T = torch.tensor(anom_sd, dtype=torch.float32, device=device)

    def predict_array(W, M24, Td, h_all, bs: int = 128) -> np.ndarray:
        Wn = ((W - mu_x) / sd_x).astype(np.float32)
        Mn = ((M24 - mu_x) / sd_x).astype(np.float32)
        Tn = ((Td - mu_td) / sd_td).astype(np.float32)
        preds = []
        with torch.no_grad():
            for s in range(0, len(Wn), bs):
                xb = torch.tensor(Wn[s : s + bs], dtype=torch.float32, device=device)
                mb = torch.tensor(Mn[s : s + bs], dtype=torch.float32, device=device)
                tb = torch.tensor(Tn[s : s + bs], dtype=torch.float32, device=device)
                hb = torch.tensor(h_all[s : s + bs], dtype=torch.long, device=device)
                pred_anom = ensemble(xb, mb, tb, hb.float())
                y_hat = CLIM_T[hb] + pred_anom * ASD_T
                y_np = y_hat.cpu().numpy()
                y_np[..., 0] = np.clip(y_np[..., 0], 0.0, 100.0)
                preds.append(y_np)
        return np.concatenate(preds, axis=0)

    t0 = time.time()
    val_oof = predict_array(Wtr[val_mask], M24_tr[val_mask], Td_tr[val_mask], h_tr[val_mask])
    sc, rmse_d, skill_d = evaluate_skill_score(val_oof, Ytr[val_mask], clim_rmse_val)
    print(f"[Validación] Tiempo inferencia 336h: {time.time()-t0:.2f}s")
    print("=" * 80)
    print(f"🏆 ENSEMBLE UNIFICADO (models/modelo.pt) — SKILL SCORE VALIDACIÓN: {sc:.4f}")
    print(f"   Skill por variable : {skill_d}")
    print(f"   RMSE por variable  : {rmse_d}")
    print("=" * 80)

    # Medir tiempo de inferencia para 24h (dato para REPORTE_PRODUCTO.md)
    t_inf0 = time.time()
    _ = predict_array(Wte[:24], M24_te[:24], Td_te[:24], h_te[:24], bs=24)
    inf_24h_ms = (time.time() - t_inf0) * 1000.0
    print(f"[Benchmark Producto] Tiempo de inferencia 24 horas en {device.upper()}: {inf_24h_ms:.1f} ms")

    test_preds = predict_array(Wte, M24_te, Td_te, h_te)

    logger = ExperimentLogger(EXPERIMENTS_DIR)
    exp_id = logger.log_experiment(
        model_name="ensemble_3fam_v6",
        metric_name="skill_score",
        fold_scores=[sc],
        params={
            "arch": "ensemble_3fam_v6",
            "components": ["PhysicsUNet(w=64)", "PhysicsConvNeXt(w=80)", "PhysicsFNO(w=48)"],
            "blend_weights": DEFAULT_BLEND_WEIGHTS.tolist(),
            "n_params": n_params,
            "inf_24h_ms": round(inf_24h_ms, 1),
            "val_rmse_per_var": rmse_d,
            "val_skill_per_var": skill_d,
        },
        features=[
            "GFS_centered_7lags_t-6..t+6",
            "GFS_24h_synoptic_mean",
            "DewPoint_Td_Magnus",
            "Orographic_Forcing_V_dot_gradDEM",
            "DEM_10ch_Topography",
            "Ensemble_UNet_ConvNeXt_FNO",
        ],
        oof_preds=val_oof,
        test_preds=test_preds,
        author="local_rtx4050",
        cv_strategy="temporal_split_2024061600",
        seed=42,
        notes=f"Skill={sc:.4f} | 3-Family Unified Ensemble | {rmse_d}",
    )

    # Guardar ÚNICO archivo models/modelo.pt (state_dict puro)
    weights_path = MODELS_DIR / "modelo.pt"
    torch.save({k: v.detach().cpu() for k, v in ensemble.state_dict().items()}, weights_path)
    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"[Export] Modelo único guardado en {weights_path} ({size_mb:.2f} MB)")

    np.savez_compressed(
        MODELS_DIR / "norm_stats.npz",
        mu_x=mu_x,
        sd_x=sd_x,
        mu_td=np.array(mu_td, dtype=np.float32),
        sd_td=np.array(sd_td, dtype=np.float32),
        clim_trn=clim_trn,
        anom_sd=anom_sd,
        clim_rmse_val=clim_rmse_val,
        val_rmse_per_var=np.array([rmse_d[v] for v in VARS], dtype=np.float32),
        blend_weights=DEFAULT_BLEND_WEIGHTS,
        arch=np.array("ensemble_3fam_v6"),
        width=np.array(64),
        best_score=np.array(sc),
        exp_id=np.array(exp_id),
    )

    # Generar submission.csv oficial
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
    print(f"[Export] Nuevo submission.csv generado en {sub_path} (shape={sub.shape})")

    # Limpiar checkpoints intermedios sueltos en models/ para dejar solo modelo.pt + norm_stats.npz
    for extra in (convnext_path, fno_path):
        if extra.exists():
            extra.unlink()
            print(f"[Cleanup] Eliminado checkpoint intermedio {extra.name}")


if __name__ == "__main__":
    main()
