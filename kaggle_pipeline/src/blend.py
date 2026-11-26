"""
Blend por variable con pesos no negativos que suman 1 (Fase 2.3 del plan), sobre el OOF
concatenado de los 3 folds.

Para cada variable v se minimiza  sum_f  s_f · || sum_j w_j P_j − y ||²   con w >= 0, sum w = 1,
donde s_f = peso_cv_f / (n_f · RMSE_clim_{f,v}²)  (fold_weighting='cv', default): cada fold aporta
como su término (RMSE/RMSE_clim)² en la métrica, ponderado 0.2/0.3/0.5. Con fold_weighting='n'
todas las muestras pesan igual (concatenación simple).

Métodos (dan el mismo óptimo; se testean entre sí):
  - 'simplex': gradiente proyectado acelerado (FISTA) sobre el simplex con la matriz de Gram MxM.
  - 'nnls'   : scipy.optimize.nnls con fila de penalización para sum=1 y renormalización.
Shrinkage: w_final = (1 − α)·w_opt + α·uniforme.

Además: evaluación por fold con kaggle_like_score y "nested" (pesos ajustados en los otros folds
y evaluados en el fold k) = estimación honesta del blend.

Uso:
  python kaggle_pipeline/src/blend.py --members convnext_42 multiscale_102 unet_203 fno_304 --shrink 0 0.3
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv import FOLD_WEIGHTS, cv_media_ponderada, fold_masks  # noqa: E402
from metrics import KAGGLE_HOURS, VARS, clim_by_hour, hours_utc, score_with_clim  # noqa: E402
from oof_io import OOF_DIR, EXPERIMENTS_DIR, load_oof_by_name  # noqa: E402


@dataclass
class FoldOOF:
    k: int
    preds: np.ndarray  # (M, N, 70, 35, 4)
    y: np.ndarray  # (N, 70, 35, 4)
    times: np.ndarray  # (N,)
    clim: np.ndarray  # (24, 70, 35, 4) climatología del train del fold


def _filter_hours(times, hours):
    return np.ones(len(times), bool) if hours is None else np.isin(hours_utc(times), np.asarray(tuple(hours)))


def make_fold_oof(k, preds, y, times, clim, hours=KAGGLE_HOURS) -> FoldOOF:
    m = _filter_hours(times, hours)
    return FoldOOF(k, np.asarray(preds, np.float32)[:, m], y[m].astype(np.float32), np.asarray(times)[m], clim)


def load_fold_oofs(names: Sequence[str], folds: Sequence[int], Ytr, ttr, oof_dir: Path = OOF_DIR, hours=KAGGLE_HOURS) -> List[FoldOOF]:
    out = []
    ttr = np.asarray(ttr).astype(str)
    for k in folds:
        trn, val = fold_masks(ttr, k)
        preds, t_ref = [], None
        for nm in names:
            p, t = load_oof_by_name(nm, k, oof_dir)
            if t_ref is None:
                t_ref = t
            elif not np.array_equal(t, t_ref):
                raise ValueError(f"times OOF de {nm} fold{k} no coinciden con {names[0]}")
            preds.append(p)
        if not np.array_equal(t_ref, ttr[val]):
            raise ValueError(f"times OOF fold{k} != val del fold según cv.py")
        out.append(make_fold_oof(k, np.stack(preds), Ytr[val], t_ref, clim_by_hour(Ytr[trn], ttr[trn]), hours))
    return out


# ---------------------------------------------------------------------------
# Optimización en el simplex
# ---------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    """Proyección euclídea sobre {w >= 0, sum w = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_simplex_qp(G: np.ndarray, b: np.ndarray, iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """min_w  w^T G w − 2 b^T w   s.t. w en el simplex (FISTA)."""
    M = len(b)
    L = 2.0 * float(np.linalg.eigvalsh(G).max()) + 1e-12
    w = np.full(M, 1.0 / M)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = project_simplex(z - (2.0 * (G @ z) - 2.0 * b) / L)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = w_new + ((t - 1) / t_new) * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def solve_nnls_sum1(G: np.ndarray, b: np.ndarray, lam: float = 1e6) -> np.ndarray:
    """Equivalente vía scipy.optimize.nnls: ||L^T w − L^{-1} b||² + lam (sum w − 1)², luego renormaliza."""
    from scipy.optimize import nnls

    M = len(b)
    scale = float(np.trace(G)) / M
    Gs, bs = G / scale, b / scale
    Lc = np.linalg.cholesky(Gs + 1e-10 * np.eye(M))
    A = np.vstack([Lc.T, np.sqrt(lam) * np.ones((1, M))])
    c = np.concatenate([np.linalg.solve(Lc, bs), [np.sqrt(lam)]])
    w, _ = nnls(A, c)
    return w / w.sum() if w.sum() > 0 else np.full(M, 1.0 / M)


def accumulate_gram(fold_oofs: Sequence[FoldOOF], use_folds: Optional[Sequence[int]] = None, fold_weighting: str = "cv"):
    """Devuelve G (4, M, M), b (4, M) acumulados sobre los folds pedidos."""
    sel = [f for f in fold_oofs if use_folds is None or f.k in use_folds]
    M = sel[0].preds.shape[0]
    G = np.zeros((4, M, M))
    b = np.zeros((4, M))
    n_tot = sum(len(f.y) for f in sel)
    for f in sel:
        h = hours_utc(f.times)
        for v in range(4):
            # Con sum(w)=1:  sum_j w_j P_j − y = sum_j w_j (P_j − y)  -> se trabaja con la Gram de
            # ERRORES (b = 0). Mucho mejor condicionada que la de predicciones crudas (T ~ 290 K).
            P = f.preds[..., v].reshape(M, -1).astype(np.float64) - f.y[..., v].reshape(1, -1).astype(np.float64)
            y = np.zeros(P.shape[1])
            if fold_weighting == "cv":
                rc2 = float(((f.clim[h][..., v].astype(np.float64) - f.y[..., v]) ** 2).mean())
                s = FOLD_WEIGHTS[f.k] / (P.shape[1] * rc2)
            elif fold_weighting == "n":
                s = 1.0 / (n_tot * P.shape[1] / len(f.y))
            else:
                raise ValueError(fold_weighting)
            G[v] += s * (P @ P.T)
            b[v] += s * (P @ y)
    return G, b


def fit_blend(fold_oofs, use_folds=None, method: str = "simplex", shrink: float = 0.0, fold_weighting: str = "cv") -> np.ndarray:
    """Pesos (M, 4) [miembro, variable], columnas suman 1 (mismo layout que EnsembleV8.blend_weights)."""
    G, b = accumulate_gram(fold_oofs, use_folds, fold_weighting)
    M = G.shape[1]
    W = np.zeros((M, 4))
    for v in range(4):
        W[:, v] = solve_simplex_qp(G[v], b[v]) if method == "simplex" else solve_nnls_sum1(G[v], b[v])
    W = (1.0 - shrink) * W + shrink / M
    return W.astype(np.float32)


def blend_predict(preds: np.ndarray, W: np.ndarray) -> np.ndarray:
    """preds (M, N, H, W, 4), W (M, 4) -> (N, H, W, 4)."""
    return np.einsum("mnhwv,mv->nhwv", preds, W, optimize=True)


def evaluate_blend(fold_oofs, W) -> Dict:
    per = {f.k: score_with_clim(blend_predict(f.preds, W), f.y, f.times, f.clim, hours=None) for f in fold_oofs}
    return {"folds": per, "cv_media": cv_media_ponderada({k: r["score"] for k, r in per.items()})}


def nested_blend_cv(fold_oofs, method="simplex", shrink=0.0, fold_weighting="cv") -> Dict:
    """Para cada fold k: pesos ajustados con los OTROS folds, evaluados en k (estimación honesta)."""
    per, weights = {}, {}
    ks = [f.k for f in fold_oofs]
    for f in fold_oofs:
        others = [k for k in ks if k != f.k]
        W = fit_blend(fold_oofs, others, method, shrink, fold_weighting)
        weights[f.k] = W
        per[f.k] = score_with_clim(blend_predict(f.preds, W), f.y, f.times, f.clim, hours=None)
    return {"folds": per, "weights": weights, "cv_media": cv_media_ponderada({k: r["score"] for k, r in per.items()})}


def _row(label, res):
    fs = " ".join(f"f{k}={r['score']:.4f}" for k, r in sorted(res["folds"].items()))
    return f"  {label:34s} {fs} | cv_media={res['cv_media']:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blend NNLS/simplex por variable sobre OOF de 3 folds")
    ap.add_argument("--members", nargs="+", required=True, help="nombres '{miembro}_{seed}' (como en oof_*_fold*.npy)")
    ap.add_argument("--folds", default="1,2,3")
    ap.add_argument("--shrink", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--method", choices=["simplex", "nnls"], default="simplex")
    ap.add_argument("--fold-weighting", choices=["cv", "n"], default="cv")
    ap.add_argument("--oof-dir", default=str(OOF_DIR))
    ap.add_argument("--data-dir", default=str(SRC_DIR.parent.parent / "data"))
    ap.add_argument("--save", default=None, help="json de salida (default experiments/blend_<members>.json)")
    a = ap.parse_args(argv)

    tr = np.load(Path(a.data_dir) / "train.npz")
    folds = [int(k) for k in a.folds.split(",")]
    fo = load_fold_oofs(a.members, folds, tr["Y"], tr["times"].astype(str), Path(a.oof_dir))
    M = len(a.members)

    print("=" * 96)
    print(f"BLEND · miembros={a.members} · folds={folds} · horas Kaggle · método={a.method}")
    for j, nm in enumerate(a.members):
        Wj = np.zeros((M, 4), np.float32)
        Wj[j] = 1.0
        print(_row(f"solo {nm}", evaluate_blend(fo, Wj)))
    print(_row("uniforme", evaluate_blend(fo, np.full((M, 4), 1.0 / M, np.float32))))
    out = {"members": a.members, "vars": VARS, "folds": folds, "results": {}}
    for s in a.shrink:
        W = fit_blend(fo, None, a.method, s, a.fold_weighting)
        r_in = evaluate_blend(fo, W)
        r_nest = nested_blend_cv(fo, a.method, s, a.fold_weighting)
        print(_row(f"opt shrink={s} (in-sample)", r_in))
        print(_row(f"opt shrink={s} (NESTED, honesto)", r_nest))
        print(f"    pesos (filas=miembros, cols={VARS}):\n" + "\n".join(f"    {nm:20s} " + " ".join(f"{x:.3f}" for x in W[j]) for j, nm in enumerate(a.members)))
        out["results"][str(s)] = {"weights": W.tolist(), "cv_in_sample": r_in["cv_media"], "cv_nested": r_nest["cv_media"],
                                  "folds_nested": {str(k): v["score"] for k, v in r_nest["folds"].items()}}
    print("=" * 96)
    save = Path(a.save) if a.save else EXPERIMENTS_DIR / f"blend_{'+'.join(a.members)}.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    with open(save, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Pesos guardados en {save}")


if __name__ == "__main__":
    main()
