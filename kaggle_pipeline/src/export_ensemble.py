"""
Exporta el ensamble final (EnsembleVN) a partir de los refits de run_cv.py --refit (Fase 3/4 del plan).

Regla de las bases: el producto debe usar EXACTAMENTE los mismos pesos que la submission final. Por eso
este script (1) arma UN módulo con los state_dicts de cada refit, (2) lo guarda en <out>/modelo.pt +
norm_stats.npz (lo que carga app/inti_inference.py), (3) lo VUELVE A CARGAR desde disco y predice el
test con él para escribir la submission, y (4) comprueba que esa predicción coincide con el blend de los
pred_test_*.npy sueltos de cada refit.

Entradas en --refit-dir (las escribe run_cv.py --refit):
  model_{miembro}_{seed}.pt, norm_{miembro}_{seed}.npz, pred_test_{miembro}_{seed}.npy (+ _times.npy)
Salidas:
  <out>/modelo.pt        state_dict de EnsembleVN. Si en float32 pesa > --fp16-threshold-mb (95 MB),
                         los PARÁMETROS se guardan en float16 (buffers en float32); al cargar se pasan a
                         float32 con .float() (ensemble_vn.load_state_dict_float). La submission se predice
                         con esos pesos ya redondeados, así que producto y submission siguen siendo idénticos.
  <out>/norm_stats.npz   stats compartidas + arch='ensemble_vn' + members_config (json) + lag_set + blend.
  <out>/model_card.json  miembros, pesos, cv_media/lb (si se pasan), sha256.
  <sub-dir>/sub_<fecha>_<id>_cv<cv>.csv  validado con check_submission.

Uso:
  python kaggle_pipeline/src/export_ensemble.py --members convnext_42 unet_203 fno-m86_304 \
      --blend experiments/blend_convnext_42+unet_203+fno-m86_304.json --refit-dir kaggle_pipeline/experiments/refit \
      --out models/ --id-exp vn3 --cv-media 0.4123 [--lb 0.501] [--submission-copy submission.csv]
  --blend uniform  -> pesos 1/N para todas las variables.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ensemble_vn import (  # noqa: E402
    build_ensemble_vn,
    load_ensemble_vn,
    member_config_from_norm,
    normalize_features,
    shared_norm_stats,
)
from metrics import VARS  # noqa: E402
from oof_io import load_pred_test  # noqa: E402
from paths import DATA_DIR, MODELS_DIR, ROOT_DIR, data_file  # noqa: E402
from train import extract_hour_utc  # noqa: E402
from train_v7 import LAG_SETS, build_v7_temporal_and_spatial_features, predict_v7  # noqa: E402

FP16_THRESHOLD_MB = 95.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_blend(blend: str, members: Sequence[str], blend_key: Optional[str] = None):
    """(W (N,4), descripción). blend = 'uniform' | json de blend.py | json {'members','weights'}."""
    n = len(members)
    if blend == "uniform":
        return np.full((n, 4), 1.0 / n, np.float32), "uniform"
    with open(blend, "r", encoding="utf-8") as f:
        js = json.load(f)
    if "results" in js:  # salida de blend.py
        if blend_key is None:  # el shrink con mejor cv_nested (estimación honesta)
            blend_key = max(js["results"], key=lambda k: js["results"][k].get("cv_nested", -np.inf))
        res = js["results"][blend_key]
        W, src = np.asarray(res["weights"], np.float32), f"{Path(blend).name} shrink={blend_key} (cv_nested={res.get('cv_nested')})"
    else:
        W, src = np.asarray(js["weights"], np.float32), Path(blend).name
    jm = list(js.get("members", members))
    if sorted(jm) != sorted(members):
        raise ValueError(f"miembros del blend {jm} != --members {list(members)}")
    W = W[[jm.index(m) for m in members]]  # reordenar al orden de --members
    if W.shape != (n, 4) or (W < -1e-6).any() or not np.allclose(W.sum(0), 1.0, atol=1e-3):
        raise ValueError(f"pesos de blend inválidos: shape={W.shape} col_sums={W.sum(0)}")
    return (W / W.sum(0, keepdims=True)).astype(np.float32), src


def load_refits(members: Sequence[str], refit_dir: Path):
    norms, cfgs, sds = [], [], []
    for nm in members:
        member, seed = nm.rsplit("_", 1)
        z = dict(np.load(refit_dir / f"norm_{member}_{seed}.npz"))
        norms.append(z)
        cfgs.append(member_config_from_norm(z, nm))
        sds.append(torch.load(refit_dir / f"model_{member}_{seed}.pt", map_location="cpu", weights_only=True))
    return norms, cfgs, sds


def state_dict_for_disk(model: torch.nn.Module, fp16_threshold_mb: float):
    """state_dict en CPU; si en float32 supera el umbral, parámetros -> float16 (buffers quedan float32)."""
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    mb32 = buf.tell() / 1e6
    if mb32 <= fp16_threshold_mb:
        return sd, "float32", mb32
    params = {k for k, _ in model.named_parameters()}
    sd = {k: (v.half() if k in params and v.is_floating_point() else v) for k, v in sd.items()}
    return sd, "float16", mb32


def predict_test(model, stats, Xte, tte, idx, device, batch_size: int = 16) -> np.ndarray:
    """Features sobre el test COMPLETO (para que los lags existan), predicción solo en `idx`."""
    lag_set = str(stats["lag_set"])
    W, M24, S24, GRAD, TD = build_v7_temporal_and_spatial_features(Xte, tte, lag_set=lag_set)
    f = normalize_features(W[idx], M24[idx], S24[idx], GRAD[idx], TD[idx], stats)
    return predict_v7(model, f["Wn"], f["Mn"], f["Sn"], f["Gn"], f["Tn"], extract_hour_utc(tte[idx]), device, bs=batch_size)


def export(members: Sequence[str], blend: str, refit_dir, out_dir, sub_dir=None, data_dir=None, id_exp: Optional[str] = None,
           cv_media: Optional[float] = None, lb: Optional[float] = None, blend_key: Optional[str] = None, tol: float = 0.05,
           fp16_threshold_mb: float = FP16_THRESHOLD_MB, device: Optional[str] = None, submission_copy=None,
           batch_size: int = 16, strict_verify: bool = True) -> Dict:
    from check_submission import build_submission, check_submission, check_submission_df
    import pandas as pd

    members = list(members)
    refit_dir, out_dir = Path(refit_dir), Path(out_dir)
    sub_dir = Path(sub_dir) if sub_dir else ROOT_DIR / "submissions"
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    id_exp = id_exp or f"vn{len(members)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) refits + stats compartidas (assert) + blend
    norms, cfgs, sds = load_refits(members, refit_dir)
    stats = shared_norm_stats(norms, members)
    lag_set = str(stats["lag_set"])
    W, blend_src = resolve_blend(blend, members, blend_key)
    print(f"[export] {len(members)} miembros | lag_set={lag_set} | blend={blend_src} | device={device}")
    for c, w in zip(cfgs, W):
        print(f"   {c['name']:28s} arch={c['arch']:10s} w={c['width']:3d} fno_modes={c['fno_modes']} "
              f"topo_extra={c.get('topo_extra', False)} wind_head={c.get('wind_head', False)} | pesos {VARS}: "
              + " ".join(f"{x:.3f}" for x in w))

    # 2) armar EnsembleVN con los state_dicts de cada refit (carga estricta)
    static = dict(np.load(data_file("static.npz", data_dir)))
    model = build_ensemble_vn(static, stats, cfgs, W, device)
    model.load_members(sds)
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))

    # 3) guardar modelo.pt (+ fp16 si hace falta) y norm_stats.npz
    sd, dtype, mb32 = state_dict_for_disk(model, fp16_threshold_mb)
    p_model, p_stats = out_dir / "modelo.pt", out_dir / "norm_stats.npz"
    torch.save(sd, p_model)
    lag_cfg = LAG_SETS[lag_set]
    np.savez_compressed(
        p_stats,
        **{k: v for k, v in stats.items() if k != "lag_set"},
        arch=np.array("ensemble_vn"),
        members_config=np.array(json.dumps(cfgs)),
        member_names=np.array(members),
        blend_weights=W,
        lag_set=np.array(lag_set), lags=np.array(lag_cfg["lags"]),
        lag_window=np.array(lag_cfg["window"]), lag_causal=np.array(lag_cfg["causal"]),
        weights_dtype=np.array(dtype),
        exp_id=np.array(id_exp),
    )
    size_mb = p_model.stat().st_size / 1e6
    print(f"[export] {p_model} ({size_mb:.2f} MB, {dtype}; float32 serían {mb32:.2f} MB) | {p_stats} | {n_params:,} params")

    # 4) RECARGAR el modelo exportado (lo mismo que hace el producto) y predecir el test con él
    model_x, stats_x = load_ensemble_vn(out_dir, static, device)
    te = np.load(data_file("test.npz", data_dir))
    Xte, tte = te["X"], te["times"].astype(str)
    sample_path = data_file("sample_submission.csv", data_dir)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    if sample is not None:
        need = set(sample["id"].str.slice(0, 10).unique())
        idx = np.array([i for i, t in enumerate(tte) if t in need], dtype=np.int64)
    else:
        idx = np.array([i for i, t in enumerate(tte) if int(t[-2:]) % 3 == 0], dtype=np.int64)
    pred = predict_test(model_x, stats_x, Xte, tte, idx, device, batch_size)
    assert not np.isnan(pred).any(), "NaN en la predicción del modelo exportado"

    # 5) verificación vs blend de los pred_test_*.npy de cada refit
    pts = []
    for nm in members:
        member, seed = nm.rsplit("_", 1)
        p, t = load_pred_test(member, int(seed), refit_dir)
        assert np.array_equal(t, tte), f"times de pred_test_{nm} != test.npz"
        pts.append(p[idx])
    ref = np.einsum("mnhwv,mv->nhwv", np.stack(pts), W, optimize=True)
    diff = {v: float(np.abs(pred[..., k] - ref[..., k]).max()) for k, v in enumerate(VARS)}
    mean_diff = {v: float(np.abs(pred[..., k] - ref[..., k]).mean()) for k, v in enumerate(VARS)}
    print(f"[export] modelo exportado vs blend(pred_test_*): max|diff| {diff} | mean|diff| {mean_diff} (tol {tol}; "
          f"pred_test está en float16)")
    if strict_verify:
        assert max(diff.values()) < tol, f"el modelo exportado NO reproduce el blend de los refits: {diff}"

    # 6) submission
    sub_path = None
    if sample is not None:
        sub = build_submission(pred, tte[idx], sample)
        errors, warns = check_submission_df(sub, sample)
        if errors:
            raise RuntimeError(f"submission inválido: {errors}")
        sub_dir.mkdir(parents=True, exist_ok=True)
        cv_tag = f"{cv_media:.4f}" if cv_media is not None else "NA"
        sub_path = sub_dir / f"sub_{datetime.now():%Y%m%d}_{id_exp}_cv{cv_tag}.csv"
        sub.to_csv(sub_path, index=False)
        assert check_submission(sub_path, sample_path), "check_submission falló"
        if submission_copy:
            Path(submission_copy).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sub_path, submission_copy)
            print(f"[export] copia -> {submission_copy}")
    else:
        print("[export] sin sample_submission.csv: no se escribe submission")

    # 7) model_card.json
    sha_model, sha_stats = sha256(p_model), sha256(p_stats)
    card_path = out_dir / "model_card.json"
    old = {}
    if card_path.exists():
        with open(card_path, "r", encoding="utf-8") as f:
            old = json.load(f)
    card = {
        "name": f"EnsembleVN ({len(members)} miembros)",
        "short_name": "EnsembleVN",
        "arch": "ensemble_vn",
        "exp_id": id_exp,
        "task": old.get("task", "Downscaling GFS 27 km (17x13) -> WRF 3 km (70x35) de RH, T, U, V"),
        "training_code": "kaggle_pipeline/src/run_cv.py --refit (por miembro) + kaggle_pipeline/src/export_ensemble.py · "
                         "reproducible en train/submission_notebook_vn.ipynb",
        "inputs": {"lag_set": lag_set, "lags_h": list(lag_cfg["lags"]), "m24_window_h": list(lag_cfg["window"]),
                   "causal": bool(lag_cfg["causal"]), "time_reference": "UTC"},
        "validation": {"cv_media": cv_media, "metric": "cv_media ponderada 0.2/0.3/0.5 de 3 folds temporales (skill vs climatología horaria)"}
        if cv_media is not None else {},
        "kaggle": {"public_lb": lb} if lb is not None else {},
        "members": [dict(c, params=int(sum(p.numel() for p in m.parameters()))) for c, m in zip(cfgs, model.members)],
        "blend_weights_order": members,
        "blend_weights_per_var": {v: [round(float(x), 6) for x in W[:, k]] for k, v in enumerate(VARS)},
        "blend_source": blend_src,
        "total_params": n_params,
        "weights": {"path": "models/modelo.pt", "size_bytes": p_model.stat().st_size, "size_mb": round(size_mb, 2),
                    "dtype": dtype, "sha256": sha_model,
                    "load_note": "float16 -> cargar con {k: v.float()} (ensemble_vn.load_state_dict_float)" if dtype == "float16" else "float32"},
        "norm_stats": {"path": "models/norm_stats.npz", "sha256": sha_stats},
        "submission": {"file": sub_path.name if sub_path else None, "max_abs_diff_vs_refit_blend": diff},
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"[export] sha256 modelo.pt={sha_model} | norm_stats.npz={sha_stats} -> {card_path}")
    return {"model": p_model, "stats": p_stats, "card": card_path, "submission": sub_path, "diff": diff, "dtype": dtype,
            "sha256": sha_model, "blend_weights": W, "members_config": cfgs}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exporta EnsembleVN (N refits) a models/ + submission predicha con el modelo exportado")
    ap.add_argument("--members", nargs="+", required=True, help="'{miembro}_{seed}' como en model_{miembro}_{seed}.pt")
    ap.add_argument("--blend", default="uniform", help="'uniform' | json de blend.py | json {'members','weights'}")
    ap.add_argument("--blend-key", default=None, help="shrink del json de blend.py (default: el de mejor cv_nested)")
    ap.add_argument("--refit-dir", required=True)
    ap.add_argument("--out", default=str(MODELS_DIR), help="carpeta de modelo.pt/norm_stats.npz/model_card.json")
    ap.add_argument("--sub-dir", default=str(ROOT_DIR / "submissions"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--id-exp", default=None)
    ap.add_argument("--cv-media", type=float, default=None)
    ap.add_argument("--lb", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.05, help="max|diff| permitido vs blend de pred_test (float16)")
    ap.add_argument("--fp16-threshold-mb", type=float, default=FP16_THRESHOLD_MB)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--submission-copy", default=None, help="copiar además la submission aquí (p. ej. submission.csv)")
    return ap


def main(argv=None) -> Dict:
    a = build_parser().parse_args(argv)
    return export(a.members, a.blend, a.refit_dir, a.out, a.sub_dir, a.data_dir, a.id_exp, a.cv_media, a.lb, a.blend_key,
                  a.tol, a.fp16_threshold_mb, a.device, a.submission_copy, a.batch_size)


if __name__ == "__main__":
    main()
