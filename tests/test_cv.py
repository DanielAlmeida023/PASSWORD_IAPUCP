from datetime import datetime

import numpy as np
import pytest

from cv import FOLDS, GAP_HOURS, assert_no_leak, cv_media_ponderada, feature_source_indices, fold_masks
from synthetic_data import hourly_times, make_dataset, make_static


@pytest.fixture(scope="module")
def times():
    return hourly_times()


def _dt(s):
    return datetime.strptime(s, "%Y%m%d%H")


def test_fold_definitions(times):
    expect = {1: ("2024051600", "2024053123"), 2: ("2024060100", "2024061523"), 3: ("2024061600", "2024063023")}
    for f in FOLDS:
        trn, val = fold_masks(times, f.k)
        assert (min(times[val]), max(times[val])) == expect[f.k]
        # train = todo lo anterior a val_start - 24 h
        assert min(times[trn]) == "2024021800"
        gap_h = (_dt(f.val_start) - _dt(max(times[trn]))).total_seconds() / 3600
        assert gap_h == GAP_HOURS + 1  # última hora de train = val_start - 25 h
        # ninguna hora del gap en train ni val
        gap = (times >= max(times[trn])) & (times < f.val_start)
        gap[np.where(times == max(times[trn]))[0]] = False
        assert gap.sum() == GAP_HOURS and not (gap & (trn | val)).any()
    assert [f.weight for f in FOLDS] == [0.2, 0.3, 0.5]


def test_assert_no_leak_detects_problems(times):
    trn, val = fold_masks(times, 3)
    bad = trn.copy()
    bad[np.where(val)[0][0] - 1] = True  # hora del gap metida en train
    with pytest.raises(AssertionError):
        assert_no_leak(times, bad, val)
    with pytest.raises(AssertionError):
        assert_no_leak(times, trn | val, val)


def test_cv_media_ponderada():
    assert cv_media_ponderada([0.40, 0.41, 0.42]) == pytest.approx(0.2 * 0.40 + 0.3 * 0.41 + 0.5 * 0.42)
    assert cv_media_ponderada({3: 0.41}) == pytest.approx(0.41)
    assert cv_media_ponderada({1: 0.30, 3: 0.40}) == pytest.approx((0.2 * 0.3 + 0.5 * 0.4) / 0.7)
    with pytest.raises(ValueError):
        cv_media_ponderada({3: 0.41}, require_all=True)


def test_val_features_read_only_neighbor_X(times):
    """Documenta: features de val leen X de horas vecinas (val/gap); nunca de > val_end + 27 h."""
    trn, val = fold_masks(times, 1)
    src = feature_source_indices(times, np.where(val)[0])
    assert max(times[src]) <= "2024060203"  # 31 may 23 + 24 h + 3 h de fallback
    assert min(times[src]) >= "2024051421"


def test_prepare_split_tensors_uses_only_train_Y():
    """Perturbar Y de val/gap NO debe cambiar nada de lo derivado de Y en prepare_split_tensors."""
    torch = pytest.importorskip("torch")
    from train_v7 import prepare_split_tensors

    t = hourly_times("2024052000", "2024063023", step_h=3)
    X, Y = make_dataset(t, seed=3)
    st = make_static(0)
    trn, val = fold_masks(t, 3)
    d1 = prepare_split_tensors(X, Y, t, st, trn, val, "cpu")
    Y2 = Y.copy()
    Y2[~trn] += np.random.default_rng(0).normal(0, 50, Y2[~trn].shape).astype(np.float32)
    d2 = prepare_split_tensors(X, Y2, t, st, trn, val, "cpu")
    for key in ["clim_trn", "Wn", "Mn", "Sn", "Gn", "Tn", "sample_weights"]:
        np.testing.assert_array_equal(d1[key], d2[key], err_msg=key)
    for key in ["clim_rmse_trn_t", "anom_sd_t", "mu_x_t", "sd_x_t", "lapse_delta_norm"]:
        assert torch.equal(d1[key], d2[key]), key
    # clim_rmse_val SÍ depende de Y de val (es el denominador de la métrica en val): esperado
    assert not np.allclose(d1["clim_rmse_val"], d2["clim_rmse_val"])
