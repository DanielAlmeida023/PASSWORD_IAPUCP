"""
Estudio offline (CPU, sin entrenar) del error de viento y del post-proceso por hora UTC (Fases 2.5 / 2.7).

Usa los OOF de experiments/oof/ (oof_io.load_oof) y la Y real de train.npz:
  1. Blend uniforme de los miembros (--members) por fold, score Kaggle-like (clim = train del fold).
  2. Diagnóstico de U y V: RMSE / sesgo por hora UTC, por banda de altura y de TPI (k=21), por fold.
  3. Post-proceso ANIDADO por hora UTC x variable (postprocess.fit_hourly_affine):
       - "bias"   : y' = p + b[h, v]
       - "affine" : y' = a[h, v] * p + b[h, v]  (regresión con shrink de a hacia 1)
       - "affine_var": a[v], b[v] sin dependencia horaria
     Parámetros estimados con los OOF de los OTROS folds (leave-one-fold-out) y, como control
     temporal estricto, solo con folds ANTERIORES ("forward": fold 1 queda sin corrección).
Salida: JSON en experiments/wind_study.json y tabla por consola.

  python kaggle_pipeline/src/wind_study.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv import FOLD_WEIGHTS, cv_media_ponderada, fold_masks  # noqa: E402
from metrics import KAGGLE_HOURS, VARS, clim_by_hour, hours_utc, score_with_clim  # noqa: E402
from oof_io import load_oof_by_name  # noqa: E402
from paths import DATA_DIR, OOF_DIR, OUT_DIR, data_file  # noqa: E402
from postprocess import apply_hourly_affine, fit_hourly_affine  # noqa: E402
from train import smooth_2d  # noqa: E402

DEFAULT_MEMBERS = "convnext_42 multiscale_102 multiscale_777 unet_42 unet_203 fno_304"


def load_blend(members: List[str], ttr: np.ndarray, Ytr: np.ndarray, oof_dir: Path) -> Dict[int, Dict]:
    out = {}
    for k in FOLD_WEIGHTS:
        trn, val = fold_masks(ttr, k)
        preds = []
        t_ref = None
        for m in members:
            p, t = load_oof_by_name(m, k, oof_dir)
            if t_ref is None:
                t_ref = t
            assert np.array_equal(t, t_ref), f"times distintos en {m} fold{k}"
            p = p.copy()
            p[..., 0] = np.clip(p[..., 0], 0.0, 100.0)
            preds.append(p)
        pos = {s: i for i, s in enumerate(ttr)}
        idx = np.array([pos[s] for s in t_ref])
        assert val[idx].all(), "OOF fuera del val del fold"
        out[k] = dict(pred=np.mean(preds, axis=0).astype(np.float32), y=Ytr[idx], t=t_ref,
                      clim=clim_by_hour(Ytr[trn], ttr[trn]))
    return out


def score(d: Dict, pred=None) -> Dict:
    return score_with_clim(d["pred"] if pred is None else pred, d["y"], d["t"], d["clim"])


def diag_wind(blend: Dict[int, Dict], dem: np.ndarray) -> Dict:
    tpi21 = dem - smooth_2d(dem, k=21)
    elev_bins = [0, 1000, 2000, 3000, 4000, 4500, 7000]
    tpi_q = np.quantile(tpi21, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    rep = {"by_hour": {}, "by_elev": {}, "by_tpi21": {}, "by_fold": {}}
    for k, d in blend.items():
        h = hours_utc(d["t"])
        km = np.isin(h, KAGGLE_HOURS)
        e = d["pred"][km] - d["y"][km]
        ec = d["clim"][h[km]] - d["y"][km]
        hk = h[km]
        rep["by_fold"][k] = {v: dict(rmse=float(np.sqrt((e[..., i] ** 2).mean())), bias=float(e[..., i].mean()),
                                     rmse_clim=float(np.sqrt((ec[..., i] ** 2).mean())),
                                     sd_pred=float(d["pred"][km][..., i].std()), sd_y=float(d["y"][km][..., i].std()))
                             for i, v in enumerate(VARS) if v in ("U", "V")}
        for hh in KAGGLE_HOURS:
            m = hk == hh
            rep["by_hour"].setdefault(hh, {})[k] = {
                v: dict(rmse=float(np.sqrt((e[m][..., i] ** 2).mean())), bias=float(e[m][..., i].mean()),
                        skill=float(1 - np.sqrt((e[m][..., i] ** 2).mean()) / np.sqrt((ec[m][..., i] ** 2).mean())))
                for i, v in ((2, "U"), (3, "V"))}
        for name, field, edges in (("by_elev", dem, elev_bins), ("by_tpi21", tpi21, tpi_q)):
            for b in range(len(edges) - 1):
                pm = (field >= edges[b]) & (field < edges[b + 1] if b < len(edges) - 2 else field <= edges[b + 1])
                if not pm.any():
                    continue
                key = f"[{edges[b]:.0f},{edges[b + 1]:.0f})"
                rep[name].setdefault(key, {"n_px": int(pm.sum())})[k] = {
                    v: dict(rmse=float(np.sqrt((e[:, pm, i] ** 2).mean())), bias=float(e[:, pm, i].mean()),
                            skill=float(1 - np.sqrt((e[:, pm, i] ** 2).mean()) / np.sqrt((ec[:, pm, i] ** 2).mean())))
                    for i, v in ((2, "U"), (3, "V"))}
    return rep


def nested_pp(blend: Dict[int, Dict], mode: str, scheme: str, shrink_a: float, vars_idx) -> Dict:
    base = {k: score(d) for k, d in blend.items()}
    res = {}
    for k, d in blend.items():
        src = [j for j in blend if j != k] if scheme == "lofo" else [j for j in blend if j < k]
        if not src:
            res[k] = dict(score=base[k]["score"], delta=0.0, skill=base[k]["skill"])
            continue
        P = np.concatenate([blend[j]["pred"] for j in src])
        Y = np.concatenate([blend[j]["y"] for j in src])
        T = np.concatenate([blend[j]["t"] for j in src])
        params = fit_hourly_affine(P, Y, T, mode=mode, shrink_a=shrink_a, vars_idx=vars_idx)
        pc = apply_hourly_affine(d["pred"], d["t"], params)
        s = score(d, pc)
        res[k] = dict(score=s["score"], delta=s["score"] - base[k]["score"], skill=s["skill"],
                      dskill={v: s["skill"][v] - base[k]["skill"][v] for v in VARS})
    res["cv_media"] = cv_media_ponderada({k: res[k]["score"] for k in blend})
    res["delta_cv_media"] = res["cv_media"] - cv_media_ponderada({k: base[k]["score"] for k in blend})
    res["all_folds_up"] = bool(all(res[k]["delta"] > 0 for k in blend))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=DEFAULT_MEMBERS)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--oof-dir", default=str(OOF_DIR))
    ap.add_argument("--out", default=str(Path(OUT_DIR) / "wind_study.json"))
    args = ap.parse_args(argv)
    members = args.members.replace(",", " ").split()

    tr = np.load(data_file("train.npz", args.data_dir))
    Ytr, ttr = tr["Y"], tr["times"].astype(str)
    dem = np.load(data_file("static.npz", args.data_dir))["dem"].astype(np.float32)
    blend = load_blend(members, ttr, Ytr, Path(args.oof_dir))

    base = {k: score(d) for k, d in blend.items()}
    out = {"members": members, "baseline": {k: dict(score=b["score"], skill=b["skill"]) for k, b in base.items()}}
    out["baseline"]["cv_media"] = cv_media_ponderada({k: b["score"] for k, b in base.items()})
    print(f"Blend uniforme {len(members)} miembros: cv_media={out['baseline']['cv_media']:.4f} | " +
          " ".join(f"f{k}={b['score']:.4f}" for k, b in base.items()))
    for k, b in base.items():
        print(f"  fold{k}: " + " ".join(f"{v}={b['skill'][v]:.4f}" for v in VARS))

    out["diag"] = diag_wind(blend, dem)
    print("\nU/V por fold (RMSE, sesgo, sd_pred/sd_y):")
    for k, r in out["diag"]["by_fold"].items():
        print(f"  fold{k}: " + " | ".join(f"{v} rmse={x['rmse']:.3f} bias={x['bias']:+.3f} clim={x['rmse_clim']:.3f} "
                                          f"sd={x['sd_pred']:.2f}/{x['sd_y']:.2f}" for v, x in r.items()))
    print("\nU/V por hora UTC (skill y sesgo, folds 1/2/3):")
    for hh, r in out["diag"]["by_hour"].items():
        print(f"  {hh:02d}z " + " | ".join(
            f"{v}: sk " + "/".join(f"{r[k][v]['skill']:.3f}" for k in r) + " b " + "/".join(f"{r[k][v]['bias']:+.2f}" for k in r)
            for v in ("U", "V")))
    for name in ("by_elev", "by_tpi21"):
        print(f"\nU/V {name} (skill f1/f2/f3):")
        for key, r in out["diag"][name].items():
            print(f"  {key:>14s} n={r['n_px']:4d} " + " | ".join(
                f"{v}: " + "/".join(f"{r[k][v]['skill']:.3f}" for k in FOLD_WEIGHTS if k in r) +
                " b " + "/".join(f"{r[k][v]['bias']:+.2f}" for k in FOLD_WEIGHTS if k in r) for v in ("U", "V")))

    out["pp"] = {}
    print("\nPost-proceso anidado (delta vs blend uniforme):")
    configs = [
        ("bias_h_all", "bias", 0.0, (0, 1, 2, 3)),
        ("bias_h_uv", "bias", 0.0, (2, 3)),
        ("affine_h_all_s0", "affine", 0.0, (0, 1, 2, 3)),
        ("affine_h_all_s05", "affine", 0.5, (0, 1, 2, 3)),
        ("affine_h_uv_s05", "affine", 0.5, (2, 3)),
        ("affine_var_all", "affine_var", 0.0, (0, 1, 2, 3)),
    ]
    for scheme in ("lofo", "forward"):
        for name, mode, sh, vi in configs:
            r = nested_pp(blend, mode, scheme, sh, vi)
            out["pp"][f"{scheme}:{name}"] = r
            print(f"  {scheme:7s} {name:18s} dcv={r['delta_cv_media']:+.4f} | " +
                  " ".join(f"f{k}={r[k]['delta']:+.4f}" for k in FOLD_WEIGHTS) +
                  " | dU=" + "/".join(f"{r[k].get('dskill', {}).get('U', 0):+.4f}" for k in FOLD_WEIGHTS) +
                  " dV=" + "/".join(f"{r[k].get('dskill', {}).get('V', 0):+.4f}" for k in FOLD_WEIGHTS))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n-> {args.out}")
    return out


if __name__ == "__main__":
    main()
