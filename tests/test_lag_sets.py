"""
Variantes de lags (FASE 5.1): el default es bit-idéntico a la implementación original y "past" es causal
(ninguna feature de la hora t depende de GFS de horas > t, incluido el relleno de bordes).
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train_v7
from train_v3 import compute_dewpoint_c
from train_v7 import LAG_SETS, V7_LAGS_PAST, build_v7_temporal_and_spatial_features

ROOT = Path(__file__).resolve().parent.parent


def _reference_v7_features(X, t):
    """Copia CONGELADA de build_v7_temporal_and_spatial_features antes de parametrizar los lags."""
    lags_ref = (-24, -6, -3, -1, 0, 1, 3, 6, 24)
    ts = pd.to_datetime(t.astype(str), format="%Y%m%d%H")
    ts_ns = ts.values.astype("datetime64[h]").astype(np.int64)
    pos = {int(h): i for i, h in enumerate(ts_ns)}
    N = len(X)
    W = np.empty((N, len(lags_ref)) + X.shape[1:], dtype=np.float32)
    M24 = np.empty_like(X, dtype=np.float32)
    S24 = np.empty_like(X, dtype=np.float32)
    for i, cur_h in enumerate(ts_ns):
        for j, lag in enumerate(lags_ref):
            target_h = int(cur_h + lag)
            idx = pos.get(target_h, None)
            if idx is None:
                for delta in (1, -1, 2, -2, 3, -3):
                    idx = pos.get(target_h + delta, None)
                    if idx is not None:
                        break
            W[i, j] = X[idx] if idx is not None else X[i]
        win_idxs = [pos[int(cur_h + d)] for d in range(-12, 13) if int(cur_h + d) in pos]
        if len(win_idxs) >= 2:
            sub = X[win_idxs]
            M24[i] = sub.mean(axis=0)
            S24[i] = sub.std(axis=0)
        else:
            M24[i] = X[i]
            S24[i] = 0.0
    rh, temp, u, v = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    dRH_row, dRH_dx = np.gradient(rh, axis=(1, 2))
    dT_row, dT_dx = np.gradient(temp, axis=(1, 2))
    dU_row, dU_dx = np.gradient(u, axis=(1, 2))
    dV_row, dV_dx = np.gradient(v, axis=(1, 2))
    GRAD = np.stack([dT_dx, -dT_row, dRH_dx, -dRH_row, dU_dx - dV_row, dV_dx + dU_row], axis=1).astype(np.float32)
    TD = compute_dewpoint_c(temp, rh)[:, None, :, :].astype(np.float32)
    return W, M24, S24, GRAD, TD


def _gappy_sequence(n=140, seed=0):
    """Serie horaria con huecos de 1..4 h y uno de 25 h (como train.npz), en unidades físicas."""
    rng = np.random.default_rng(seed)
    steps = np.ones(n - 1, dtype=int)
    steps[[10, 40, 41, 70, 100]] = [2, 3, 4, 25, 2]
    start = datetime(2024, 5, 1)
    hours = np.concatenate([[0], np.cumsum(steps)])
    t = np.array([(start + timedelta(hours=int(h))).strftime("%Y%m%d%H") for h in hours])
    X = np.empty((n, 17, 13, 4), dtype=np.float32)
    X[..., 0] = rng.uniform(30, 100, (n, 17, 13))
    X[..., 1] = rng.uniform(265, 300, (n, 17, 13))
    X[..., 2:] = rng.normal(0, 3, (n, 17, 13, 2))
    return X, t


def _assert_bit_identical(a, b):
    assert len(a) == len(b) == 5
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(x, y)


def test_default_bit_identical_to_original_synthetic():
    X, t = _gappy_sequence()
    ref = _reference_v7_features(X, t)
    _assert_bit_identical(build_v7_temporal_and_spatial_features(X, t), ref)
    _assert_bit_identical(build_v7_temporal_and_spatial_features(X, t, lag_set="default"), ref)


def test_default_bit_identical_to_original_real_test():
    p = ROOT / "data" / "test.npz"
    if not p.exists():
        pytest.skip("sin data/test.npz")
    d = np.load(p)
    X, t = d["X"][:200], d["times"].astype(str)[:200]
    _assert_bit_identical(build_v7_temporal_and_spatial_features(X, t), _reference_v7_features(X, t))


def test_lag_set_invariants():
    assert train_v7.V7_LAGS == (-24, -6, -3, -1, 0, 1, 3, 6, 24)
    assert LAG_SETS["default"]["window"] == (-12, 12) and not LAG_SETS["default"]["causal"]
    past = LAG_SETS["past"]
    assert past["lags"] == V7_LAGS_PAST and past["causal"] and past["window"] == (-24, 0)
    assert len(V7_LAGS_PAST) == 9 and len(set(V7_LAGS_PAST)) == 9 and max(V7_LAGS_PAST) == 0
    # slots leídos por índice en las arquitecturas: 0 = t-24, 2 = t-3, 4 = t
    assert V7_LAGS_PAST[0] == -24 and V7_LAGS_PAST[2] == -3 and V7_LAGS_PAST[4] == 0
    with pytest.raises(ValueError):
        build_v7_temporal_and_spatial_features(*_gappy_sequence(), lag_set="nope")


def test_past_is_causal_truncation_invariant():
    """Las features de la hora k no cambian si se borra todo el GFS posterior a k (huecos incluidos)."""
    X, t = _gappy_sequence()
    full = build_v7_temporal_and_spatial_features(X, t, lag_set="past")
    for k in [0, 1, 5, 11, 12, 30, 41, 42, 71, 72, 101, len(t) - 1]:
        trunc = build_v7_temporal_and_spatial_features(X[: k + 1], t[: k + 1], lag_set="past")
        for a, b in zip(full, trunc):
            assert np.array_equal(a[k], b[k]), f"hora {k} depende del futuro"


def test_past_sources_never_after_t_and_exact_inside():
    X, t = _gappy_sequence()
    n = len(t)
    off = 1000.0
    X_idx = np.ascontiguousarray(np.broadcast_to((np.arange(n, dtype=np.float32) + off)[:, None, None, None], (n, 2, 2, 4)))
    W = build_v7_temporal_and_spatial_features(X_idx, t, lag_set="past")[0]
    src = np.rint(W[:, :, 0, 0, 1].astype(np.float64) - off).astype(int)
    hrs = pd.to_datetime(t, format="%Y%m%d%H").values.astype("datetime64[h]").astype(np.int64)
    got = hrs[src] - hrs[:, None]
    assert (got <= 0).all()
    # tramo contiguo final (>=24 h tras el hueco de 25 h): todos los lags exactos
    assert (got[-5:] == np.array(V7_LAGS_PAST)).all()


def test_run_cv_lags_flag_and_member_name():
    import run_cv

    a = run_cv.build_parser().parse_args(["--model", "unet"])
    assert a.lags == "default" and run_cv._resolve_hparams(a)["member"] == "unet"
    b = run_cv.build_parser().parse_args(["--model", "unet", "--lags", "past", "--tag", "x"])
    assert run_cv._resolve_hparams(b)["member"] == "unet-x-lagpast"


def test_inference_reads_lag_set_from_stats():
    sys.path.insert(0, str(ROOT / "app"))
    import inti_inference as ii

    assert ii.lag_set_from_stats(None) == "default"
    assert ii.lag_set_from_stats({"mu_x": np.zeros(4)}) == "default"
    assert ii.lag_set_from_stats({"lag_set": np.array("past")}) == "past"
    X, t = _gappy_sequence()
    W_past = ii.build_features(X, t, lag_set="past")[0]
    assert np.array_equal(W_past, build_v7_temporal_and_spatial_features(X, t, lag_set="past")[0])
    assert ii.lag_report(t, lag_set="past")["lags_h"] == list(V7_LAGS_PAST)


def test_run_cv_smoke_past_refit_saves_lag_config(tmp_path):
    pytest.importorskip("torch")
    import run_cv

    run_cv.main(["--model", "fno", "--smoke", "--refit", "--lags", "past", "--fno-modes", "6,4", "--out-dir", str(tmp_path)])
    z = np.load(tmp_path / "refit" / "norm_fno-lagpast_304.npz")
    assert str(z["lag_set"]) == "past" and tuple(z["lags"]) == V7_LAGS_PAST and tuple(z["lag_window"]) == (-24, 0)
