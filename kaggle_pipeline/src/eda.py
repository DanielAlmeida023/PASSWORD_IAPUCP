"""
Análisis Exploratorio de Datos (EDA) — Hackathon IA PUCP 2026 (AIRE-ANDINO).
Analiza:
1. Huecos temporales en train y test.
2. Deriva estacional (Febrero húmedo -> Junio/Julio/Agosto seco) en GFS (X) y WRF (Y).
3. Relación espacial GFS vs. WRF y diagnóstico de errores del baseline/TopoResUNet.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
VARS = ["RH", "T", "U", "V"]


def main():
    tr = np.load(DATA_DIR / "train.npz")
    te = np.load(DATA_DIR / "test.npz")
    st = np.load(DATA_DIR / "static.npz")
    Xtr, Ytr, ttr = tr["X"], tr["Y"], tr["times"].astype(str)
    Xte, tte = te["X"], te["times"].astype(str)
    dem = st["dem"]

    ts_tr = pd.to_datetime(ttr, format="%Y%m%d%H")
    ts_te = pd.to_datetime(tte, format="%Y%m%d%H")

    print("=== 1. HUECOS TEMPORALES ===")
    diffs_tr = ts_tr.to_series().diff().dt.total_seconds().div(3600).dropna()
    diffs_te = ts_te.to_series().diff().dt.total_seconds().div(3600).dropna()
    print(f"Train: {len(ttr)} horas ({ttr[0]} -> {ttr[-1]}) | Huecos > 1h: {int((diffs_tr > 1).sum())} (máx {diffs_tr.max():.0f}h)")
    print(f"Test : {len(tte)} horas ({tte[0]} -> {tte[-1]}) | Huecos > 1h: {int((diffs_te > 1).sum())} (máx {diffs_te.max():.0f}h)")

    print("\n=== 2. DERIVA ESTACIONAL MES A MES (GFS vs WRF) ===")
    for m in sorted(ts_tr.month.unique()):
        mask = ts_tr.month == m
        xm = Xtr[mask].mean(axis=(0, 1, 2))
        ym = Ytr[mask].mean(axis=(0, 1, 2))
        print(
            f"Train Mes {m:02d} (n={mask.sum():4d}) | "
            f"GFS: RH={xm[0]:5.1f} T={xm[1]:6.2f} U={xm[2]:5.2f} V={xm[3]:5.2f} | "
            f"WRF: RH={ym[0]:5.1f} T={ym[1]:6.2f} U={ym[2]:5.2f} V={ym[3]:5.2f}"
        )

    for m in sorted(ts_te.month.unique()):
        mask = ts_te.month == m
        xm = Xte[mask].mean(axis=(0, 1, 2))
        print(
            f"Test  Mes {m:02d} (n={mask.sum():4d}) | "
            f"GFS: RH={xm[0]:5.1f} T={xm[1]:6.2f} U={xm[2]:5.2f} V={xm[3]:5.2f}"
        )

    print("\n=== 3. COMPARACIÓN CLIMATOLOGÍA OFICIAL (sample_submission) ===")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print("Media sample_submission:", {v: round(float(sub[v].mean()), 3) for v in VARS})
    print("Media Ytr (Feb-Jun)    :", {v: round(float(Ytr[..., i].mean()), 3) for i, v in enumerate(VARS)})
    val_m = ttr >= "2024061600"
    print("Media Ytr (16-30 Jun)  :", {v: round(float(Ytr[val_m, ..., i].mean()), 3) for i, v in enumerate(VARS)})

    print("\n=== 4. RANGOS FÍSICOS EN Ytr Y DEM ===")
    for i, v in enumerate(VARS):
        print(f"  {v:2s}: min={Ytr[..., i].min():7.2f} | p01={np.percentile(Ytr[..., i], 1):7.2f} | p99={np.percentile(Ytr[..., i], 99):7.2f} | max={Ytr[..., i].max():7.2f}")
    print(f"  DEM: min={dem.min():.1f} m | median={np.median(dem):.1f} m | max={dem.max():.1f} m")

    # Correlación GFS vs WRF y diagnóstico de errores del último TopoResUNet si existe OOF
    oof_candidates = sorted((ROOT_DIR / "kaggle_pipeline" / "experiments").glob("*toporesunet*/oof_preds.npy"))
    if oof_candidates:
        oof = np.load(oof_candidates[-1])
        y_val = Ytr[val_m]
        err = oof - y_val
        print(f"\n=== 5. DIAGNÓSTICO DE ERRORES ({oof_candidates[-1].parent.name}) ===")
        bands = [(0, 1500, "Costa/Bajo (<1500m)"), (1500, 3500, "Valles (1500-3500m)"), (3500, 4500, "Puna (3500-4500m)"), (4500, 6000, "Alta Montaña (>4500m)")]
        for lo, hi, label in bands:
            bmask = (dem >= lo) & (dem < hi)
            rmses = {v: round(float(np.sqrt((err[:, bmask, i] ** 2).mean())), 3) for i, v in enumerate(VARS)}
            print(f"  {label:24s} ({bmask.sum():4d} px): RMSE = {rmses}")

        hours_val = np.array([int(k[-2:]) for k in ttr[val_m]])
        print("  RMSE por bloque horario (Hora Perú = UTC-5):")
        for h_lo, h_hi, lbl in [(0, 6, "Madrugada (00-06h)"), (6, 12, "Mañana (06-12h)"), (12, 18, "Tarde (12-18h)"), (18, 24, "Noche (18-24h)")]:
            h_pe = (hours_val - 5) % 24
            hmask = (h_pe >= h_lo) & (h_pe < h_hi)
            rmses = {v: round(float(np.sqrt((err[hmask, ..., i] ** 2).mean())), 3) for i, v in enumerate(VARS)}
            print(f"    {lbl:20s}: RMSE = {rmses}")


if __name__ == "__main__":
    main()
