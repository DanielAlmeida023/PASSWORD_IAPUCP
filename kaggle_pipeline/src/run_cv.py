"""
CV temporal de 3 folds (y refit 100 %) para UN miembro de la suite v8 (Fases 1 y 2 del plan).

Modo CV (default), por cada fold (cv.FOLDS, gap 24 h):
  1. prepare_split_tensors(...) con las máscaras del fold -> normalización, climatología,
     anom_sd y pesos de muestra calculados SOLO con el train del fold.
  2. train_member_v7(...) (mismo loop de entrenamiento que train_v8 / notebook).
  3. OOF del val en float16 (offset T, ver oof_io.py) en OUT_DIR/oof/.
  4. Score con metrics.kaggle_like_score (horas 00,03,..,21; clim = train del fold).
  Al final: fila en OUT_DIR/registro.csv + resumen JSON junto al OOF.

Modo --refit: entrena con el 100 % del train por --epochs (o --steps de optimizador) fijos,
  SIN selección por val (estado EMA final), predice test y guarda en OUT_DIR/refit/:
  pred_test_{miembro}_{seed}.npy (float16 con offset, leer con oof_io.load_pred_test),
  model_{miembro}_{seed}.pt (state_dict), norm_{miembro}_{seed}.npz (stats de normalización)
  y, si hay sample_submission.csv, sub_{fecha}_refit_{miembro}_{seed}.csv validado.

Rutas: PASSWORD_DATA_DIR (Kaggle: /kaggle/input/hackathon-ia-pucp-2026) y PASSWORD_OUT_DIR
(Kaggle: /kaggle/working); fallback <repo>/data y <repo>/kaggle_pipeline/experiments (paths.py).

Ejemplos:
  python kaggle_pipeline/src/run_cv.py --model unet --seed 203 --epochs 24
  python kaggle_pipeline/src/run_cv.py --model fno --seed 304 --fno-modes 8,6 --wd 4e-2 --tag m86
  python kaggle_pipeline/src/run_cv.py --model convnext --loss-w-uv 1.3 --recency-weights plan --tag uv13
  python kaggle_pipeline/src/run_cv.py --model unet --seed 203 --refit --epochs 16
  python kaggle_pipeline/src/run_cv.py --model unet --smoke              # sintético, 1 época, CPU
  python kaggle_pipeline/src/run_cv.py --model unet --smoke --smoke-real # data real submuestreada
  python kaggle_pipeline/src/run_cv.py --model unet --lags past          # plan B causal (solo GFS <= t)

--lags past (FASE 5.1): lags train_v7.V7_LAGS_PAST y ventana M24/S24 [t-24, t], relleno causal. El miembro
recibe el sufijo "-lagpast" (OOF, summary, registro, refit) para no mezclarlo con el default, y el refit
guarda lag_set/lags/lag_window en norm_*.npz para que la inferencia use la misma variante.

Opciones de viento (FASE 2.5; por defecto apagadas -> comportamiento idéntico al histórico):
  --topo-extra : +9 canales topo del DEM (topo_extra.py: TPI k=21, curvaturas, aspecto suavizado, ejes del
                 valle estimado por PCA). Sufijo de miembro "-topox".
  --wind-head  : cabeza de viento U,V más profunda (train_v8.DeepWindHead). Sufijo "-wh".
  El refit guarda topo_extra/wind_head (+ valley_axis_deg) en norm_*.npz; ensemble_vn.member_config_from_norm
  los pasa a members_config y EnsembleVN / app/inti_inference.py reconstruyen el miembro igual.
"""

import argparse
import json
import math
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ensemble_vn import build_member_model  # noqa: E402
from cv import FOLD_WEIGHTS, cv_media_ponderada, fold_masks, get_fold  # noqa: E402
from metrics import VARS, format_score, kaggle_like_score  # noqa: E402
from oof_io import save_oof, save_pred_test  # noqa: E402
from paths import DATA_DIR, OUT_DIR, data_file  # noqa: E402
from registro import append_registro, git_commit_short  # noqa: E402
from seed import seed_everything  # noqa: E402
from train import smooth_2d  # noqa: E402
from train_v7 import (  # noqa: E402
    LAG_SETS,
    build_v7_temporal_and_spatial_features,
    predict_v7,
    prepare_split_tensors,
    train_member_v7,
)
from train_v8 import DirectResConvNeXtV8, DirectResFNOV8, DirectResMultiScaleV8, DirectResUNetV8  # noqa: E402
from topo_extra import build_topo_extra_features  # noqa: E402

# Hiperparámetros por defecto = los de train_v8.main / notebook (EnsembleV8 actual)
MODEL_DEFAULTS: Dict[str, Dict] = {
    "convnext": dict(cls=DirectResConvNeXtV8, w=84, dropout=0.15, lr=1.8e-3, wd=1.5e-2, noise=0.045, seed=42),
    "multiscale": dict(cls=DirectResMultiScaleV8, w=96, dropout=0.18, lr=1.6e-3, wd=1.8e-2, noise=0.050, seed=102),
    "unet": dict(cls=DirectResUNetV8, w=64, dropout=0.14, lr=1.5e-3, wd=1.4e-2, noise=0.040, seed=203),
    "fno": dict(cls=DirectResFNOV8, w=32, dropout=0.18, lr=1.8e-3, wd=2.5e-2, noise=0.050, seed=304),
}

RECENCY_PRESETS: Dict[str, Optional[Dict[int, float]]] = {
    "v8": {2: 0.50, 3: 0.80, 4: 0.90, 5: 1.15, 6: 1.45},  # el de prepare_split_tensors (actual)
    "plan": {2: 0.4, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0},  # Fase 2.4 del plan
    "uniform": None,
}


def parse_recency(spec: str) -> Optional[Dict[int, float]]:
    if spec in RECENCY_PRESETS:
        return RECENCY_PRESETS[spec]
    out = {}
    for tok in spec.split(","):
        m, w = tok.split(":")
        out[int(m)] = float(w)
    return out


def recency_sample_weights(times: np.ndarray, trn_mask: np.ndarray, month_map: Optional[Dict[int, float]]) -> np.ndarray:
    months = pd.to_datetime(np.asarray(times).astype(str), format="%Y%m%d%H").month.values
    if month_map is None:
        sw = np.ones(len(times), dtype=np.float32)
    else:
        sw = np.array([month_map.get(int(m), 1.0) for m in months], dtype=np.float32)
    sw[trn_mask] /= sw[trn_mask].mean()
    return sw


def build_model(name: str, d: Dict, dem_delta_km_t: torch.Tensor, w: int, dropout: float, fno_modes=None,
                topo_extra_t: Optional[torch.Tensor] = None, wind_head: bool = False):
    # Definición única en ensemble_vn.build_member_model (la misma que usa EnsembleVN en la inferencia).
    # OJO: fno_modes / topo_extra / wind_head cambian las formas del state_dict -> el ensamble debe
    # construirse con la misma config (se guarda en norm_*.npz del refit).
    return build_member_model(name, d["grid_t"], d["topo_t"], d["lapse_delta_norm"], dem_delta_km_t,
                              d["mu_x_t"], d["sd_x_t"], d["anom_sd_t"], w=w, dropout=dropout, fno_modes=fno_modes,
                              topo_extra=topo_extra_t, wind_head=wind_head)


def topo_extra_tensor(args, st: Dict, device: str):
    """(tensor (K,70,35) | None, info del eje del valle | {}) según --topo-extra."""
    if not args.topo_extra:
        return None, {}
    feats, info = build_topo_extra_features(st["dem"], st["wrf_lats"], st["wrf_lons"])
    print(f"[run_cv] topo-extra: +{feats.shape[0]} canales | eje del valle rumbo={info['bearing_deg']:.1f}° "
          f"(n={info['n_cells']} celdas, elongación={info['elongation']:.1f})")
    return torch.tensor(feats, dtype=torch.float32, device=device), info


def load_data(args, need_test: bool = False):
    """Devuelve Xtr, Ytr, ttr, st, (Xte, tte) | (None, None)."""
    if args.smoke and not args.smoke_real:
        from synthetic_data import hourly_times, make_dataset, make_static

        ttr = hourly_times(step_h=3)  # 3 h -> todas horas Kaggle, ~1000 muestras
        Xtr, Ytr = make_dataset(ttr, seed=0)
        st = make_static(seed=0)
        Xte = tte = None
        if need_test:
            tte = hourly_times("2024070200", "2024080423", step_h=3)
            Xte, _ = make_dataset(tte, seed=1)
        return Xtr, Ytr, ttr, st, Xte, tte
    tr = np.load(data_file("train.npz", args.data_dir))
    st = dict(np.load(data_file("static.npz", args.data_dir)))
    Xte = tte = None
    if need_test:
        te = np.load(data_file("test.npz", args.data_dir))
        Xte, tte = te["X"], te["times"].astype(str)
    return tr["X"], tr["Y"], tr["times"].astype(str), st, Xte, tte


def _resolve_hparams(args) -> Dict:
    dflt = MODEL_DEFAULTS[args.model]
    return dict(
        seed=dflt["seed"] if args.seed is None else args.seed,
        w=args.width or dflt["w"],
        dropout=dflt["dropout"] if args.dropout is None else args.dropout,
        lr=args.lr or dflt["lr"],
        wd=args.wd or dflt["wd"],
        noise=dflt["noise"] if args.noise_std is None else args.noise_std,
        fno_modes=tuple(int(x) for x in args.fno_modes.split(",")) if args.fno_modes else None,
        month_map=parse_recency(args.recency_weights),
        loss_w=[1.0, 1.0, args.loss_w_uv, args.loss_w_uv] if args.loss_w_uv != 1.0 else None,
        topo_extra=bool(args.topo_extra),
        wind_head=bool(args.wind_head),
        member=args.model + (f"-{args.tag}" if args.tag else "") + ("" if args.lags == "default" else f"-lag{args.lags}")
        + ("-topox" if args.topo_extra else "") + ("-wh" if args.wind_head else ""),
    )


def _subsample(mask: np.ndarray, n: int, seed: int, times: Optional[np.ndarray] = None) -> np.ndarray:
    """Submuestreo para debug/smoke. Con `times`, estratifica por hora UTC (clim sin horas vacías)."""
    if not n or mask.sum() <= n:
        return mask
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    if times is None:
        keep = rng.choice(idx, size=n, replace=False)
    else:
        h = np.array([int(str(t)[-2:]) for t in np.asarray(times)[idx]])
        groups = [idx[h == k] for k in np.unique(h)]
        per = max(1, n // len(groups))
        keep = np.concatenate([rng.choice(g, size=min(per, len(g)), replace=False) for g in groups])
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def _describe(args, hp, epochs) -> str:
    return (
        f"{hp['member']} s{hp['seed']} ep{epochs} w{hp['w']} do{hp['dropout']} lr{hp['lr']} wd{hp['wd']} noise{hp['noise']}"
        + (f" fno{hp['fno_modes']}" if hp["fno_modes"] else "")
        + (f" lossUV{args.loss_w_uv}" if hp["loss_w"] else "")
        + f" rec={args.recency_weights}"
        + (" noValSel" if args.no_val_selection else "")
        + (f" stop{args.stop_epoch}" if getattr(args, "stop_epoch", 0) else "")
        + f" lags={args.lags}"
        + (" topoExtra" if hp["topo_extra"] else "")
        + (" windHead" if hp["wind_head"] else "")
    )


def run(args) -> Dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    hp = _resolve_hparams(args)
    seed, member = hp["seed"], hp["member"]
    out_dir = Path(args.out_dir)
    oof_dir = out_dir / "oof"
    registro_path = Path(args.registro) if args.registro else out_dir / "registro.csv"
    folds = [int(k) for k in args.folds.split(",")]

    print(f"[run_cv] {member} seed={seed} | device={device} | folds={folds} | epochs={args.epochs} | "
          f"w={hp['w']} do={hp['dropout']} lr={hp['lr']} wd={hp['wd']} noise={hp['noise']} fno_modes={hp['fno_modes']} "
          f"loss_w={hp['loss_w']} recency={args.recency_weights} select_on_val={not args.no_val_selection} lags={args.lags} "
          f"topo_extra={hp['topo_extra']} wind_head={hp['wind_head']} | out={out_dir}")

    Xtr, Ytr, ttr, st, _, _ = load_data(args)
    dem = st["dem"].astype(np.float32)
    dem_delta_km_t = torch.tensor(((dem - smooth_2d(dem, k=9)) / 1000.0).astype(np.float32), device=device)
    topo_extra_t, _ = topo_extra_tensor(args, st, device)

    fold_res = {}
    for k in folds:
        t0 = time.time()
        trn_mask, val_mask = fold_masks(ttr, k)
        trn_mask = _subsample(trn_mask, args.max_train, seed, ttr)  # solo smoke/debug
        val_mask = _subsample(val_mask, args.max_val, seed)
        d = prepare_split_tensors(Xtr, Ytr, ttr, st, trn_mask, val_mask, device, lag_set=args.lags)
        sw = recency_sample_weights(ttr, trn_mask, hp["month_map"])

        seed_everything(seed)  # init de pesos reproducible por (modelo, seed)
        model = build_model(args.model, d, dem_delta_km_t, hp["w"], hp["dropout"], hp["fno_modes"],
                            topo_extra_t, hp["wind_head"])
        _, oof, _, best_ep, _, _ = train_member_v7(
            f"{member} s{seed} fold{k}", model, trn_mask, val_mask,
            d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
            sw, d["clim_rmse_trn_t"], d["clim_rmse_val"], device,
            epochs=args.epochs, batch_size=args.batch_size, lr=hp["lr"], weight_decay=hp["wd"], noise_std=hp["noise"],
            seed=seed, loss_weights=hp["loss_w"], select_on_val=not args.no_val_selection, stop_epoch=args.stop_epoch,
        )
        assert not np.isnan(oof).any(), f"NaN en OOF {member} fold{k}"
        p = save_oof(oof, ttr[val_mask], member, seed, k, oof_dir)
        res = kaggle_like_score(oof, Ytr[val_mask], ttr[val_mask], Ytr[trn_mask], ttr[trn_mask])
        res.update(best_ep=int(best_ep), n_train=int(trn_mask.sum()), n_val=int(val_mask.sum()), secs=round(time.time() - t0, 1), oof=str(p))
        fold_res[k] = res
        print(f"[run_cv] fold{k}: {format_score(res)} | best_ep={best_ep} | {res['secs']}s -> {p}")
        del d, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_media = cv_media_ponderada({k: r["score"] for k, r in fold_res.items()})
    skill_media = {v: cv_media_ponderada({k: r["skill"][v] for k, r in fold_res.items()}) for v in VARS}
    print(f"[run_cv] {member} s{seed}: cv_media ponderada={cv_media:.4f} | " + " ".join(f"{v}={s:.4f}" for v, s in skill_media.items()))

    summary = {
        "member": member, "seed": seed, "model": args.model, "epochs": args.epochs,
        "hparams": {k: v for k, v in hp.items() if k != "month_map"} | {"recency": args.recency_weights, "batch_size": args.batch_size,
                                                                       "select_on_val": not args.no_val_selection},
        "folds": {str(k): r for k, r in fold_res.items()}, "cv_media": cv_media, "skill_media": skill_media,
        "fold_weights": FOLD_WEIGHTS, "smoke": bool(args.smoke),
        "lags": {"lag_set": args.lags, **{k: list(v) if isinstance(v, tuple) else v for k, v in LAG_SETS[args.lags].items()}},
    }
    oof_dir.mkdir(parents=True, exist_ok=True)
    with open(oof_dir / f"oof_{member}_{seed}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.no_registro:
        partial = "" if set(folds) == set(FOLD_WEIGHTS) else f" [solo folds {args.folds}; cv_media renormalizada]"
        append_registro({
            "id_exp": args.id_exp or f"cv_{member}_s{seed}" + ("_SMOKE" if args.smoke else ""),
            "descripcion": (args.descripcion or "CV " + _describe(args, hp, args.epochs)) + partial,
            "cv_fold1": fold_res.get(1, {}).get("score"),
            "cv_fold2": fold_res.get(2, {}).get("score"),
            "cv_fold3": fold_res.get(3, {}).get("score"),
            "cv_media": cv_media,
            **{f"skill_{v}": skill_media[v] for v in VARS},
            "commit": git_commit_short(),
        }, registro_path)
        print(f"[run_cv] fila añadida a {registro_path}")
    return summary


def run_refit(args) -> Dict:
    """Entrena con el 100 % del train (épocas/pasos fijos, sin mirar val) y predice test."""
    from check_submission import build_submission, check_submission_df

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    hp = _resolve_hparams(args)
    seed, member = hp["seed"], hp["member"]
    out_dir = Path(args.out_dir)
    refit_dir = out_dir / "refit"
    refit_dir.mkdir(parents=True, exist_ok=True)
    registro_path = Path(args.registro) if args.registro else out_dir / "registro.csv"

    Xtr, Ytr, ttr, st, Xte, tte = load_data(args, need_test=True)
    trn_mask = _subsample(np.ones(len(ttr), dtype=bool), args.max_train, seed, ttr)
    # "val" solo para log por época (IN-SAMPLE, no es validación): últimas 2 semanas de junio
    log_mask = _subsample(ttr >= get_fold(3).val_start, args.max_val or 96, seed)
    n_batches = int(math.ceil(trn_mask.sum() / args.batch_size))
    epochs = max(1, int(math.ceil(args.steps / n_batches))) if args.steps else args.epochs
    print(f"[refit] {member} seed={seed} | device={device} | N_train={trn_mask.sum()} | epochs={epochs} "
          f"({epochs * n_batches} pasos) | out={refit_dir}")

    dem = st["dem"].astype(np.float32)
    dem_delta_km_t = torch.tensor(((dem - smooth_2d(dem, k=9)) / 1000.0).astype(np.float32), device=device)
    d = prepare_split_tensors(Xtr, Ytr, ttr, st, trn_mask, log_mask, device, lag_set=args.lags)
    sw = recency_sample_weights(ttr, trn_mask, hp["month_map"])
    topo_extra_t, topo_info = topo_extra_tensor(args, st, device)
    seed_everything(seed)
    model = build_model(args.model, d, dem_delta_km_t, hp["w"], hp["dropout"], hp["fno_modes"], topo_extra_t, hp["wind_head"])
    t0 = time.time()
    state, _, _, _, _, _ = train_member_v7(
        f"REFIT {member} s{seed}", model, trn_mask, log_mask,
        d["Wn"], d["Mn"], d["Sn"], d["Gn"], d["Tn"], d["h_tr"], Ytr,
        sw, d["clim_rmse_trn_t"], d["clim_rmse_val"], device,
        epochs=epochs, batch_size=args.batch_size, lr=hp["lr"], weight_decay=hp["wd"], noise_std=hp["noise"],
        seed=seed, loss_weights=hp["loss_w"], select_on_val=False, stop_epoch=args.stop_epoch,
    )
    model = build_model(args.model, d, dem_delta_km_t, hp["w"], 0.0, hp["fno_modes"], topo_extra_t, hp["wind_head"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})

    # Features de test construidas SOLO con X de test (igual que el notebook) y normalizadas con stats del train
    W, M24, S24, GRAD, TD = build_v7_temporal_and_spatial_features(Xte, tte, lag_set=args.lags)
    pred = predict_v7(
        model,
        ((W - d["mu_x"]) / d["sd_x"]).astype(np.float32),
        ((M24 - d["mu_x"]) / d["sd_x"]).astype(np.float32),
        (S24 / d["sd_x"]).astype(np.float32),
        ((GRAD - d["mu_grad"]) / d["sd_grad"]).astype(np.float32),
        ((TD - d["mu_td"]) / d["sd_td"]).astype(np.float32),
        np.array([int(t[-2:]) for t in tte], dtype=np.int64), device,
    )
    assert not np.isnan(pred).any(), "NaN en predicción de test"

    p_pred = save_pred_test(pred, tte, member, seed, refit_dir)
    p_model = refit_dir / f"model_{member}_{seed}.pt"
    torch.save({k: v.detach().cpu() for k, v in state.items()}, p_model)
    np.savez_compressed(
        refit_dir / f"norm_{member}_{seed}.npz",
        mu_x=d["mu_x"], sd_x=d["sd_x"], mu_td=np.array(d["mu_td"], np.float32), sd_td=np.array(d["sd_td"], np.float32),
        mu_grad=d["mu_grad"], sd_grad=d["sd_grad"], clim_trn=d["clim_trn"], anom_sd=d["anom_sd"],
        arch=np.array(args.model), width=np.array(hp["w"]), epochs=np.array(epochs),
        fno_modes=np.array(hp["fno_modes"] or (12, 8)),
        # config de lags: la inferencia (app/inti_inference.py) la lee; ausente => "default"
        lag_set=np.array(args.lags), lags=np.array(LAG_SETS[args.lags]["lags"]),
        lag_window=np.array(LAG_SETS[args.lags]["window"]), lag_causal=np.array(LAG_SETS[args.lags]["causal"]),
        # opciones de arquitectura (ensemble_vn.member_config_from_norm; ausentes => False)
        topo_extra=np.array(hp["topo_extra"]), wind_head=np.array(hp["wind_head"]),
        **({"valley_axis_deg": np.array(topo_info["bearing_deg"], np.float32)} if topo_info else {}),
    )
    print(f"[refit] listo en {time.time() - t0:.1f}s -> {p_pred}, {p_model}")

    sub_name = ""
    sample_path = data_file("sample_submission.csv", args.data_dir)
    if not (args.smoke and not args.smoke_real) and sample_path.exists():
        sample = pd.read_csv(sample_path)
        sub = build_submission(pred, tte, sample)
        errors, warns = check_submission_df(sub, sample)
        if errors:
            print("[refit] SUBMISSION INVÁLIDO:", errors)
        else:
            sub_dir = out_dir / "submissions"
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_name = f"sub_{datetime.now():%Y%m%d}_refit_{member}_{seed}.csv"
            sub.to_csv(sub_dir / sub_name, index=False)
            print(f"[refit] submission válido -> {sub_dir / sub_name}" + (f" | avisos: {warns}" if warns else ""))

    if not args.no_registro:
        append_registro({
            "id_exp": args.id_exp or f"refit_{member}_s{seed}" + ("_SMOKE" if args.smoke else ""),
            "descripcion": args.descripcion or f"REFIT 100% {_describe(args, hp, epochs)} ({epochs * n_batches} pasos)",
            "archivo_submission": sub_name,
            "commit": git_commit_short(),
        }, registro_path)
    return {"pred_test": str(p_pred), "model": str(p_model), "submission": sub_name, "epochs": epochs}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CV temporal 3 folds / refit 100 % para un miembro v8")
    p.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    p.add_argument("--seed", type=int, default=None, help="default = semilla del miembro en train_v8")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--stop-epoch", type=int, default=0, help="parar tras K épocas de un calendario de --epochs (0 = no)")
    p.add_argument("--steps", type=int, default=0, help="(refit) nº de pasos de optimizador; si >0 define las épocas")
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--refit", action="store_true", help="entrenar 100%% train y predecir test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None, help="weight decay")
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--fno-modes", default=None, help="'m1,m2' para las capas FNO (solo --model fno), p. ej. 8,6")
    p.add_argument("--loss-w-uv", type=float, default=1.0, help="peso relativo de U y V en la pérdida (RH=T=1)")
    p.add_argument("--recency-weights", default="v8", help="v8 | plan | uniform | '2:0.4,3:0.4,4:0.6,5:0.8,6:1.0'")
    p.add_argument("--no-val-selection", action="store_true", help="CV: no elegir época con val (OOF honesto, última época EMA)")
    p.add_argument("--lags", choices=sorted(LAG_SETS), default="default",
                   help="default = lags ±24 h + M24/S24 [t-12,t+12]; past = causal (solo GFS <= t), M24/S24 [t-24,t]")
    p.add_argument("--topo-extra", action="store_true",
                   help="+9 canales topo del DEM (TPI k=21, curvatura, aspecto k5, ejes del valle) [sufijo -topox]")
    p.add_argument("--wind-head", action="store_true",
                   help="cabeza de viento U,V más profunda (2 conv 3x3 dilatadas residuales) [sufijo -wh]")
    p.add_argument("--tag", default="", help="sufijo del miembro en nombres de archivos (p. ej. m86)")
    p.add_argument("--id-exp", default=None)
    p.add_argument("--descripcion", default=None)
    p.add_argument("--data-dir", default=str(DATA_DIR), help="default PASSWORD_DATA_DIR o <repo>/data")
    p.add_argument("--out-dir", default=None, help="default PASSWORD_OUT_DIR o <repo>/kaggle_pipeline/experiments")
    p.add_argument("--registro", default=None, help="default <out-dir>/registro.csv")
    p.add_argument("--no-registro", action="store_true")
    p.add_argument("--max-train", type=int, default=0, help="submuestrear train (debug)")
    p.add_argument("--max-val", type=int, default=0, help="submuestrear val (debug)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 época, CPU, train submuestreado, salidas en carpeta temporal")
    p.add_argument("--smoke-real", action="store_true", help="con --smoke: usar data real en vez de sintética")
    return p


def main(argv=None) -> Dict:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.epochs, args.steps, args.cpu = 1, 0, True
        args.max_train = args.max_train or 96
        if args.smoke_real:
            args.max_val = args.max_val or 64
        if args.out_dir is None:
            args.out_dir = tempfile.mkdtemp(prefix="run_cv_smoke_")
    if args.out_dir is None:
        args.out_dir = str(OUT_DIR)
    return run_refit(args) if args.refit else run(args)


if __name__ == "__main__":
    main()
