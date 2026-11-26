"""
EnsembleVN: ensamble de N miembros heterogéneos (varias semillas/arquitecturas/fno_modes) en UN nn.Module.

Generaliza train_v8.EnsembleV8 (4 miembros fijos) para el envío final (Fase 3/4 del plan):
  - `members` : nn.ModuleList, un DirectRes*V8 por miembro, construido con su propia config
                (arch, width, fno_modes) y dropout=0 (inferencia).
  - `blend_weights` : buffer (N, 4) [miembro, variable RH/T/U/V], columnas >= 0 que suman 1
                (mismo layout que blend.fit_blend y EnsembleV8.blend_weights).
  - forward: RH de CADA miembro se recorta a [0, 100] antes de mezclar, igual que predict_v7 hace con
    cada refit (pred_test_*.npy) y con cada OOF sobre el que blend.py ajusta los pesos. Así el modelo
    exportado reproduce exactamente blend(pred_test_*) (el recorte final de predict_v7 queda inocuo).

Normalización y lags COMPARTIDOS: todos los refits de run_cv.py --refit usan el 100 % del train, así que
prepare_split_tensors da las mismas stats (mu_x, sd_x, mu/sd_td, mu/sd_grad, anom_sd, clim) para todos;
`shared_norm_stats` lo verifica con assert y también que todos usen el mismo lag_set (las features de
entrada se construyen una sola vez para todo el ensamble).

La config de cada miembro se lee del norm_{miembro}_{seed}.npz del refit (arch, width, fno_modes,
lag_set). Se guarda como json en models/norm_stats.npz (`members_config`) para reconstruir el módulo
en la inferencia (app/inti_inference.py) sin depender de los archivos del refit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from train_v7 import FNOBlock2D
from train_v8 import DirectResConvNeXtV8, DirectResFNOV8, DirectResMultiScaleV8, DirectResUNetV8

ARCH_CLASSES = {
    "convnext": DirectResConvNeXtV8,
    "multiscale": DirectResMultiScaleV8,
    "unet": DirectResUNetV8,
    "fno": DirectResFNOV8,
}
DEFAULT_FNO_MODES = (12, 8)  # el de DirectResFNOV8: capas (12,8),(12,8),(8,6)

# Claves de normalización que deben ser idénticas entre miembros (las guarda run_cv.run_refit)
SHARED_STAT_KEYS = ("mu_x", "sd_x", "mu_td", "sd_td", "mu_grad", "sd_grad", "anom_sd", "clim_trn")


def build_member_model(arch: str, grid, topo, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd,
                       w: int, dropout: float, fno_modes: Optional[Sequence[int]] = None,
                       topo_extra: Optional[torch.Tensor] = None, wind_head: bool = False) -> nn.Module:
    """
    Construye un miembro DirectRes*V8 (única definición; run_cv.build_model la usa).
    topo_extra : None (default) o tensor (K, 70, 35) de topo_extra.build_topo_extra_features -> se concatena
                 detrás de los 10 canales topo (run_cv --topo-extra; cambia formas del state_dict).
    wind_head  : cabeza de viento DeepWindHead (run_cv --wind-head; cambia el state_dict).
    """
    if arch not in ARCH_CLASSES:
        raise ValueError(f"arquitectura desconocida {arch!r}; opciones {sorted(ARCH_CLASSES)}")
    if topo_extra is not None:
        topo = torch.cat([topo, topo_extra.to(device=topo.device, dtype=topo.dtype)], dim=0)
    kw = {"wind_head": True} if wind_head else {}
    model = ARCH_CLASSES[arch](grid, topo, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd, w=w, dropout=dropout, **kw)
    if arch == "fno" and fno_modes is not None:
        m1, m2 = (int(x) for x in fno_modes)
        # capas 1-2 con (m1, m2); capa 3 con (min(8,m1), min(6,m2)) como el original (12,8),(12,8),(8,6).
        # Cambia las formas del state_dict -> el ensamble debe construirse con los mismos modes.
        model.fno_layers = nn.Sequential(
            FNOBlock2D(w, m1, m2, dropout),
            FNOBlock2D(w, m1, m2, dropout),
            FNOBlock2D(w, min(8, m1), min(6, m2), dropout),
        )
    return model


def _scalar(z, key, default=None):
    if key not in z:
        return default
    v = np.asarray(z[key])
    return v.item() if v.ndim == 0 else v.tolist()


def member_config_from_norm(norm: Dict[str, np.ndarray], name: str) -> Dict:
    """Config de inferencia de un miembro a partir de su norm_{miembro}_{seed}.npz."""
    arch = str(np.asarray(norm["arch"]))
    member, seed = name.rsplit("_", 1)
    fno_modes = [int(x) for x in np.asarray(norm["fno_modes"]).tolist()] if arch == "fno" and "fno_modes" in norm else None
    return {
        "name": name,
        "member": member,
        "seed": int(seed),
        "arch": arch,
        "width": int(np.asarray(norm["width"])),
        "fno_modes": fno_modes,
        "lag_set": str(np.asarray(norm["lag_set"])) if "lag_set" in norm else "default",
        "epochs": _scalar(norm, "epochs"),
        # opciones de arquitectura (Fase 2.5); ausentes en refits antiguos => False
        "topo_extra": bool(_scalar(norm, "topo_extra", False)),
        "wind_head": bool(_scalar(norm, "wind_head", False)),
    }


def shared_norm_stats(norms: Sequence[Dict[str, np.ndarray]], names: Sequence[str], rtol: float = 1e-5, atol: float = 1e-6) -> Dict[str, np.ndarray]:
    """
    Verifica (assert) que todos los miembros tienen las mismas stats de normalización y el mismo lag_set,
    y devuelve las del primero. Con refits al 100 % del train las stats son bit a bit iguales; la
    tolerancia solo absorbe diferencias de reducción entre máquinas (Kaggle vs local).
    """
    ref = norms[0]
    lag0 = str(np.asarray(ref["lag_set"])) if "lag_set" in ref else "default"
    for nm, z in zip(names[1:], norms[1:]):
        lag = str(np.asarray(z["lag_set"])) if "lag_set" in z else "default"
        assert lag == lag0, f"lag_set distinto: {names[0]}={lag0!r} vs {nm}={lag!r} (el ensamble comparte features)"
        for k in SHARED_STAT_KEYS:
            a, b = np.asarray(ref[k], np.float64), np.asarray(z[k], np.float64)
            assert a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol), (
                f"stats '{k}' distintas entre {names[0]} y {nm} (max|diff|="
                f"{np.abs(a - b).max() if a.shape == b.shape else 'shape'}): ¿algún refit no usó el 100 % del train "
                f"(--max-train/--smoke con otra semilla)? Una sola normalización compartida exige stats idénticas."
            )
    out = {k: np.asarray(ref[k]) for k in SHARED_STAT_KEYS}
    out["lag_set"] = np.array(lag0)
    return out


def normalize_features(W, M24, S24, GRAD, TD, stats) -> Dict[str, np.ndarray]:
    """Normalización de `train_v7.prepare_split_tensors` con stats guardadas (= run_cv.run_refit en test)."""
    mu_x, sd_x = stats["mu_x"], stats["sd_x"]
    mu_grad, sd_grad = stats["mu_grad"], stats["sd_grad"]
    mu_td, sd_td = float(stats["mu_td"]), float(stats["sd_td"])
    return {
        "Wn": ((W - mu_x) / sd_x).astype(np.float32),
        "Mn": ((M24 - mu_x) / sd_x).astype(np.float32),
        "Sn": (S24 / sd_x).astype(np.float32),
        "Gn": ((GRAD - mu_grad) / sd_grad).astype(np.float32),
        "Tn": ((TD - mu_td) / sd_td).astype(np.float32),
    }


def build_context(static: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Tensores compartidos (grid, topo, lapse, dem_delta, mu/sd/anom_sd) igual que prepare_split_tensors."""
    from train import build_sampling_grid, build_topographic_features, smooth_2d

    dem = static["dem"].astype(np.float32)
    dem_smooth = smooth_2d(dem, k=9)
    lapse_phys = (-0.0065 * (dem - dem_smooth)).astype(np.float32)
    sd_x = np.asarray(stats["sd_x"], np.float32)
    from topo_extra import build_topo_extra_features

    topo_extra, _ = build_topo_extra_features(dem, static["wrf_lats"], static["wrf_lons"])
    return {
        "grid": build_sampling_grid(static, device),
        "topo": torch.tensor(build_topographic_features(dem, static["wrf_lats"], static["wrf_lons"]), dtype=torch.float32, device=device),
        "lapse_delta_norm": torch.tensor(lapse_phys / float(sd_x[1]), dtype=torch.float32, device=device),
        "dem_delta_km": torch.tensor(((dem - dem_smooth) / 1000.0).astype(np.float32), device=device),
        "topo_extra": torch.tensor(topo_extra, dtype=torch.float32, device=device),
        "mu_x": torch.tensor(np.asarray(stats["mu_x"], np.float32), device=device),
        "sd_x": torch.tensor(sd_x, device=device),
        "anom_sd": torch.tensor(np.asarray(stats["anom_sd"], np.float32), device=device),
    }


def _member_topo_extra(cfg: Dict, topo_extra: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not cfg.get("topo_extra", False):
        return None
    if topo_extra is None:
        raise ValueError(f"el miembro {cfg.get('name')} usa topo_extra pero no se pasó el tensor (build_context lo da)")
    return topo_extra


class EnsembleVN(nn.Module):
    """N miembros DirectRes heterogéneos + pesos de mezcla por variable (N, 4), salida física (B, 70, 35, 4)."""

    def __init__(self, grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd,
                 members_config: List[Dict], blend_weights: np.ndarray, topo_extra: Optional[torch.Tensor] = None):
        super().__init__()
        bw = np.asarray(blend_weights, dtype=np.float32)
        n = len(members_config)
        assert n >= 1, "EnsembleVN necesita al menos 1 miembro"
        assert bw.shape == (n, 4), f"blend_weights {bw.shape} != ({n}, 4)"
        assert (bw >= -1e-6).all() and np.allclose(bw.sum(0), 1.0, atol=1e-4), f"blend_weights inválidos (col sums {bw.sum(0)})"
        self.members_config = [dict(c) for c in members_config]
        self.members = nn.ModuleList([
            build_member_model(c["arch"], grid, topo_features, lapse_delta_norm, dem_delta_km, mu_x, sd_x, anom_sd,
                               w=int(c["width"]), dropout=0.0, fno_modes=c.get("fno_modes"),
                               topo_extra=_member_topo_extra(c, topo_extra), wind_head=bool(c.get("wind_head", False)))
            for c in members_config
        ])
        self.register_buffer("blend_weights", torch.tensor(bw, dtype=torch.float32))

    def forward(self, w_norm, m24_norm, s24_norm, grad_norm, td_norm, h):
        out = None
        for i, m in enumerate(self.members):
            y = m(w_norm, m24_norm, s24_norm, grad_norm, td_norm, h)
            # recorte de RH por miembro (= predict_v7 en cada refit/OOF) antes de mezclar
            y = torch.cat([y[..., :1].clamp(0.0, 100.0), y[..., 1:]], dim=-1)
            term = self.blend_weights[i].view(1, 1, 1, 4) * y
            out = term if out is None else out + term
        return out

    def load_members(self, state_dicts: Sequence[Dict[str, torch.Tensor]]) -> None:
        """Carga estricta del state_dict de cada refit en su miembro."""
        assert len(state_dicts) == len(self.members)
        for c, m, sd in zip(self.members_config, self.members, state_dicts):
            dev = next(m.parameters()).device
            m.load_state_dict({k: v.float().to(dev) if v.is_floating_point() else v.to(dev) for k, v in sd.items()}, strict=True)


def build_ensemble_vn(static: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], members_config: List[Dict],
                      blend_weights: np.ndarray, device: str = "cpu") -> EnsembleVN:
    ctx = build_context(static, stats, device)
    return EnsembleVN(ctx["grid"], ctx["topo"], ctx["lapse_delta_norm"], ctx["dem_delta_km"], ctx["mu_x"], ctx["sd_x"],
                      ctx["anom_sd"], members_config, blend_weights, topo_extra=ctx["topo_extra"]).to(device)


def load_state_dict_float(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Lee models/modelo.pt. Si se exportó en float16 (archivo > 95 MB en float32), los tensores flotantes
    se pasan a float32 con .float() aquí; load_state_dict también castea, pero así es explícito.
    """
    sd = torch.load(path, map_location=device, weights_only=True)
    return {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}


def load_ensemble_vn(models_dir: Path, static: Dict[str, np.ndarray], device: str = "cpu"):
    """(modelo en eval, stats) desde <models_dir>/modelo.pt + norm_stats.npz (arch='ensemble_vn')."""
    models_dir = Path(models_dir)
    stats = dict(np.load(models_dir / "norm_stats.npz"))
    arch = str(stats["arch"])
    assert arch == "ensemble_vn", f"norm_stats.npz declara arch={arch!r}, se esperaba 'ensemble_vn'"
    cfg = json.loads(str(stats["members_config"]))
    model = build_ensemble_vn(static, stats, cfg, stats["blend_weights"], device)
    model.load_state_dict(load_state_dict_float(models_dir / "modelo.pt", device), strict=True)
    model.eval()
    return model, stats
