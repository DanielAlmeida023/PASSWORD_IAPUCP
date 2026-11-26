"""
CV temporal rolling-origin de 3 folds (Fase 1.1 del plan).

    Fold 1: val = 16–31 may   · train = todo < 16 may 00 UTC − GAP
    Fold 2: val = 01–15 jun   · train = todo < 01 jun 00 UTC − GAP
    Fold 3: val = 16–30 jun   · train = todo < 16 jun 00 UTC − GAP   (≈ holdout actual)
    Pesos cv_media ponderada: 0.2 / 0.3 / 0.5 (fold 3 = más parecido al test jul–ago).

GAP = 24 h: las horas en [val_start − 24 h, val_start) NO entran ni a train ni a val.

Sobre fugas y los lags ±24 h (V7_LAGS = (-24,-6,-3,-1,0,1,3,6,24) + ventana M24/S24 de ±12 h):
- Las features de CUALQUIER muestra se construyen solo con X (GFS) de horas vecinas, nunca con Y.
  Una muestra de val puede leer X de horas de val/gap/train y una de train (cerca del borde) puede
  leer X del gap o de val (lag +24). Eso es aceptable: X del test también está disponible completo
  en inferencia (Kaggle entrega test.npz entero), así que no es información del futuro "objetivo".
- Todo lo derivado de Y (climatología, anom_sd, clim_rmse para la pérdida, sample_weights) se calcula
  SOLO con la máscara de train del fold (ver train_v7.prepare_split_tensors; verificado en
  tests/test_cv.py perturbando Y de val/gap).
- El GAP evita que el modelo vea Y de las horas inmediatamente anteriores a val (fuerte
  autocorrelación de Y a < 24 h) y que una muestra de train comparta exactamente la misma ventana
  de lags X con la primera muestra de val.
- La normalización de X (mu_x/sd_x, grad, td) también se calcula solo con train del fold.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

GAP_HOURS = 24


@dataclass(frozen=True)
class Fold:
    k: int
    val_start: str  # 'YYYYMMDDHH' inclusivo
    val_end: str  # 'YYYYMMDDHH' inclusivo
    weight: float


FOLDS: Tuple[Fold, ...] = (
    Fold(1, "2024051600", "2024053123", 0.2),
    Fold(2, "2024060100", "2024061523", 0.3),
    Fold(3, "2024061600", "2024063023", 0.5),
)
FOLD_WEIGHTS: Dict[int, float] = {f.k: f.weight for f in FOLDS}


def get_fold(k: int) -> Fold:
    for f in FOLDS:
        if f.k == int(k):
            return f
    raise KeyError(f"Fold {k} no existe (válidos: {[f.k for f in FOLDS]})")


def _to_dt(t) -> datetime:
    return datetime.strptime(str(t), "%Y%m%d%H")


def _shift(t: str, hours: int) -> str:
    return (_to_dt(t) + timedelta(hours=hours)).strftime("%Y%m%d%H")


def fold_masks(times: Sequence, fold: Union[int, Fold], gap_hours: int = GAP_HOURS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (trn_mask, val_mask) booleanos de len(times).
    train = t < val_start − gap_hours ; val = val_start <= t <= val_end.
    (Comparación lexicográfica válida porque 'YYYYMMDDHH' tiene ancho fijo.)
    """
    f = get_fold(fold) if not isinstance(fold, Fold) else fold
    t = np.asarray(times).astype(str)
    trn_end_excl = _shift(f.val_start, -int(gap_hours))
    trn = t < trn_end_excl
    val = (t >= f.val_start) & (t <= f.val_end)
    assert_no_leak(t, trn, val, gap_hours)
    return trn, val


def assert_no_leak(times: Sequence, trn_mask: np.ndarray, val_mask: np.ndarray, gap_hours: int = GAP_HOURS) -> None:
    """Verifica: sin solapamiento, train estrictamente anterior a val y separación >= gap_hours."""
    t = np.asarray(times).astype(str)
    if (trn_mask & val_mask).any():
        raise AssertionError("train y val se solapan")
    if not trn_mask.any() or not val_mask.any():
        raise AssertionError("train o val vacío")
    last_trn = _to_dt(max(t[trn_mask]))
    first_val = _to_dt(min(t[val_mask]))
    sep_h = (first_val - last_trn).total_seconds() / 3600.0
    # la última hora de train es < val_start - gap, por lo que la separación es > gap (>= gap+1 h)
    if sep_h <= gap_hours:
        raise AssertionError(f"Separación train→val de {sep_h:.0f} h <= gap {gap_hours} h")


def all_fold_masks(times: Sequence, folds: Optional[Sequence[int]] = None, gap_hours: int = GAP_HOURS):
    ks = [f.k for f in FOLDS] if folds is None else [int(k) for k in folds]
    return {k: fold_masks(times, k, gap_hours) for k in ks}


def cv_media_ponderada(
    scores: Union[Mapping[int, float], Sequence[Optional[float]]],
    weights: Optional[Mapping[int, float]] = None,
    require_all: bool = False,
) -> float:
    """
    scores: {fold_k: score} o lista [s1, s2, s3] (None = fold no corrido).
    Con folds faltantes renormaliza los pesos sobre los disponibles (salvo require_all=True).
    """
    w = dict(FOLD_WEIGHTS if weights is None else weights)
    if not isinstance(scores, Mapping):
        scores = {i + 1: s for i, s in enumerate(scores)}
    avail = {int(k): float(s) for k, s in scores.items() if s is not None and not np.isnan(s)}
    if require_all and set(avail) != set(w):
        raise ValueError(f"Faltan folds: {sorted(set(w) - set(avail))}")
    if not avail:
        return float("nan")
    tot = sum(w[k] for k in avail)
    return float(sum(w[k] * s for k, s in avail.items()) / tot)


def feature_source_indices(times: Sequence, idx: np.ndarray, lags=(-24, -6, -3, -1, 0, 1, 3, 6, 24), window: int = 12) -> np.ndarray:
    """
    Índices de `times` cuyo X podría leerse al construir features (lags + ventana ±window) de las
    muestras `idx`. Solo para auditoría/documentación: muestra que val lee X (nunca Y) de vecinos.
    Incluye el fallback ±1..3 h de build_v7_temporal_and_spatial_features.
    """
    t = np.asarray(times).astype(str)
    hrs = np.array([int(_to_dt(s).timestamp() // 3600) for s in t])
    pos = {h: i for i, h in enumerate(hrs)}
    out = set()
    for i in np.asarray(idx):
        h0 = hrs[i]
        cands = [h0 + lag + d for lag in lags for d in (0, 1, -1, 2, -2, 3, -3)]
        cands += [h0 + d for d in range(-window, window + 1)]
        out.update(pos[c] for c in cands if c in pos)
    return np.array(sorted(out), dtype=np.int64)


def describe_folds(times: Sequence, gap_hours: int = GAP_HOURS) -> List[str]:
    t = np.asarray(times).astype(str)
    lines = []
    for f in FOLDS:
        trn, val = fold_masks(t, f, gap_hours)
        lines.append(
            f"Fold {f.k} (w={f.weight}): train {min(t[trn])}–{max(t[trn])} ({trn.sum()} h) | "
            f"gap {gap_hours} h | val {min(t[val])}–{max(t[val])} ({val.sum()} h)"
        )
    return lines


if __name__ == "__main__":
    from synthetic_data import hourly_times

    for line in describe_folds(hourly_times()):
        print(line)
