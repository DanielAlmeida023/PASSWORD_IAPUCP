"""
Features topográficas EXTRA en la grilla WRF 3 km (70, 35), calculadas SOLO del DEM de static.npz (Fase 2.5).

Opción `run_cv.py --topo-extra`. Se concatenan DESPUÉS de los 10 canales de
`train.build_topographic_features` (que ya trae TPI k=3, TPI k=9 y aspecto de la pendiente cruda), así que
los canales 1/2 (dy, dx) que usa `oro_feats` no se mueven. Canales extra (todos normalizados):
   0. TPI k=21 (~63 km): valle inter-andino vs cordillera (completa el TPI multiescala 3/9/21)
   1. Curvatura fina: -laplaciano del DEM suavizado k=3 (>0 cresta/convexo, <0 fondo de valle)
   2. Curvatura mesoescala: -laplaciano del DEM suavizado k=9
   3. sin(aspecto) del DEM suavizado k=5 (orientación de ladera a mesoescala, menos ruidosa)
   4. cos(aspecto) del DEM suavizado k=5
   5. Pendiente a lo largo del eje del valle (gradiente suavizado . eje)
   6. Pendiente transversal al eje del valle (gradiente suavizado . normal)
   7. Distancia firmada al eje del valle (km, + hacia el este del eje)
   8. Coordenada a lo largo del eje (km, + hacia el norte)

Eje del valle (Callejón de Huaylas) ESTIMADO del DEM, no fijado a mano:
  celdas candidatas = TPI k=21 en el 20 % más bajo y DEM > 1500 m (excluye la llanura costera, que no es
  un valle inter-andino); se toma la mayor componente 8-conexa y se hace PCA (ponderada por la
  profundidad -TPI) de sus coordenadas en km. El primer componente es el eje (sentido: hacia el norte).
  En el DEM oficial sale NNO–SSE (rumbo ≈ 330°), coherente con el Callejón de Huaylas.

Determinista: depende solo del DEM y las coordenadas, así que el export y el producto la recalculan
igual; `valley_axis_deg` se guarda además en norm_*.npz / members_config para trazabilidad.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np

from train import smooth_2d

N_TOPO_BASE = 10
TOPO_EXTRA_NAMES = (
    "tpi21", "curv_k3", "curv_k9", "sin_aspect_k5", "cos_aspect_k5",
    "slope_along_valley", "slope_cross_valley", "dist_valley_axis", "coord_along_valley",
)
N_TOPO_EXTRA = len(TOPO_EXTRA_NAMES)


def _z(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def _km_coords(wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    LAT, LON = np.meshgrid(np.asarray(wrf_lats, np.float64), np.asarray(wrf_lons, np.float64), indexing="ij")
    lat0 = float(np.mean(wrf_lats))
    x = (LON - LON.mean()) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (LAT - LAT.mean()) * 110.57
    return x, y


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab = np.zeros(mask.shape, np.int32)
    best, best_n, cur = 0, 0, 0
    H, W = mask.shape
    for i0, j0 in zip(*np.where(mask)):
        if lab[i0, j0]:
            continue
        cur += 1
        lab[i0, j0] = cur
        q, n = deque([(i0, j0)]), 0
        while q:
            i, j = q.popleft()
            n += 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not lab[a, b]:
                        lab[a, b] = cur
                        q.append((a, b))
        if n > best_n:
            best, best_n = cur, n
    return lab == best


def estimate_valley_axis(dem: np.ndarray, wrf_lats: np.ndarray, wrf_lons: np.ndarray,
                         tpi_quantile: float = 0.2, min_elev: float = 1500.0) -> Dict:
    """Eje principal del valle inter-andino por PCA de las celdas bajas (relativas) del DEM."""
    dem = dem.astype(np.float32)
    tpi21 = dem - smooth_2d(dem, k=21)
    cand = (tpi21 <= np.quantile(tpi21, tpi_quantile)) & (dem > min_elev)
    if cand.sum() < 10:  # DEM sintético/plano: cae a todas las celdas bajas
        cand = tpi21 <= np.quantile(tpi21, tpi_quantile)
    comp = _largest_component(cand)
    x, y = _km_coords(wrf_lats, wrf_lons)
    wgt = np.maximum(-tpi21[comp], 1e-3).astype(np.float64)
    P = np.stack([x[comp], y[comp]], 1)
    c = (P * wgt[:, None]).sum(0) / wgt.sum()
    C = ((P - c) * wgt[:, None]).T @ (P - c) / wgt.sum()
    evals, evecs = np.linalg.eigh(C)
    ax = evecs[:, -1]
    if ax[1] < 0:
        ax = -ax
    bearing = float(np.degrees(np.arctan2(ax[0], ax[1])) % 360.0)  # desde el norte, sentido horario
    return {"axis": ax.astype(np.float64), "center_km": c, "bearing_deg": bearing, "n_cells": int(comp.sum()),
            "elongation": float(np.sqrt(evals[-1] / max(evals[0], 1e-9))), "mask": comp}


def build_topo_extra_features(dem: np.ndarray, wrf_lats: np.ndarray, wrf_lons: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """(N_TOPO_EXTRA, 70, 35) float32 normalizados + info del eje del valle."""
    dem = dem.astype(np.float32)
    tpi21 = dem - smooth_2d(dem, k=21)
    curv3 = -_laplacian(smooth_2d(dem, k=3))
    curv9 = -_laplacian(smooth_2d(dem, k=9))
    d5 = smooth_2d(dem, k=5)
    dy5, dx5 = np.gradient(d5)  # fila 0 = sur -> dy hacia el norte, dx hacia el este
    asp = np.arctan2(dy5, dx5)
    info = estimate_valley_axis(dem, wrf_lats, wrf_lons)
    ax_e, ax_n = info["axis"]
    nr_e, nr_n = ax_n, -ax_e  # normal: eje rotado -90° (apunta al este si el eje apunta al norte)
    x, y = _km_coords(wrf_lats, wrf_lons)
    xc, yc = x - info["center_km"][0], y - info["center_km"][1]
    feats = np.stack([
        _z(tpi21), _z(curv3), _z(curv9),
        np.sin(asp).astype(np.float32), np.cos(asp).astype(np.float32),
        _z(dx5 * ax_e + dy5 * ax_n), _z(dx5 * nr_e + dy5 * nr_n),
        _z(xc * nr_e + yc * nr_n), _z(xc * ax_e + yc * ax_n),
    ]).astype(np.float32)
    assert feats.shape[0] == N_TOPO_EXTRA and np.isfinite(feats).all()
    return feats, {k: v for k, v in info.items() if k != "mask"}


def _laplacian(a: np.ndarray) -> np.ndarray:
    p = np.pad(a.astype(np.float64), 1, mode="edge")
    return (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * p[1:-1, 1:-1]).astype(np.float32)
