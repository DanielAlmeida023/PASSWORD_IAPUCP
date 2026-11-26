import numpy as np
import pytest

from metrics import KAGGLE_HOURS, clim_by_hour, kaggle_hour_mask, kaggle_like_score, score_with_clim
from synthetic_data import hourly_times, make_dataset


@pytest.fixture(scope="module")
def data():
    t = hourly_times("2024060100", "2024063023")  # 720 h
    X, Y = make_dataset(t, seed=1)
    trn = t < "2024061600"
    return t, Y, trn, ~trn


def test_clim_by_hour_shape_and_values(data):
    t, Y, trn, _ = data
    clim = clim_by_hour(Y[trn], t[trn])
    assert clim.shape == (24, 70, 35, 4) and clim.dtype == np.float32
    h = np.array([int(s[-2:]) for s in t[trn]])
    np.testing.assert_allclose(clim[5], Y[trn][h == 5].mean(0), rtol=1e-5, atol=1e-4)


def test_clim_missing_hour_is_nan():
    t = np.array(["2024060100", "2024060103"])
    Y = np.ones((2, 70, 35, 4), np.float32)
    clim = clim_by_hour(Y, t)
    assert np.isnan(clim[1]).all() and not np.isnan(clim[0]).any()


def test_prediction_equal_climatology_scores_zero(data):
    t, Y, trn, val = data
    clim = clim_by_hour(Y[trn], t[trn])
    h = np.array([int(s[-2:]) for s in t[val]])
    res = kaggle_like_score(clim[h], Y[val], t[val], Y[trn], t[trn])
    assert abs(res["score"]) < 1e-6
    for v in res["skill"].values():
        assert abs(v) < 1e-6


def test_perfect_prediction_scores_one(data):
    t, Y, trn, val = data
    res = kaggle_like_score(Y[val].copy(), Y[val], t[val], Y[trn], t[trn])
    assert res["score"] == pytest.approx(1.0)
    assert all(r == pytest.approx(0.0) for r in res["rmse"].values())


def test_only_kaggle_hours_are_evaluated(data):
    t, Y, trn, val = data
    clim = clim_by_hour(Y[trn], t[trn])
    pred = Y[val].copy()
    km = kaggle_hour_mask(t[val])
    pred[~km] += 1000.0  # error enorme solo en horas NO evaluadas por Kaggle
    res = kaggle_like_score(pred, Y[val], t[val], Y[trn], t[trn])
    assert res["score"] == pytest.approx(1.0)
    assert res["n_eval"] == int(km.sum()) == len(t[val]) * len(KAGGLE_HOURS) // 24
    res_all = score_with_clim(pred, Y[val], t[val], clim, hours=None)
    assert res_all["score"] < 0


def test_matches_manual_formula(data):
    t, Y, trn, val = data
    rng = np.random.default_rng(0)
    pred = Y[val] + rng.normal(0, 1.0, Y[val].shape).astype(np.float32)
    clim = clim_by_hour(Y[trn], t[trn])
    m = kaggle_hour_mask(t[val])
    h = np.array([int(s[-2:]) for s in t[val]])[m]
    yv = Y[val][m].astype(np.float64)
    rmse = np.sqrt(((pred[m] - yv) ** 2).mean(axis=(0, 1, 2)))
    rmse_c = np.sqrt(((clim[h] - yv) ** 2).mean(axis=(0, 1, 2)))
    expected = 1 - np.mean(rmse / rmse_c)
    res = kaggle_like_score(pred, Y[val], t[val], Y[trn], t[trn])
    assert res["score"] == pytest.approx(expected, abs=1e-6)


def test_raises_if_eval_hour_missing_in_clim():
    t_ref = np.array([f"20240601{h:02d}" for h in range(0, 24, 6)])  # sin 03, 09, ...
    Yr = np.ones((len(t_ref), 70, 35, 4), np.float32)
    t = np.array(["2024060203"])
    with pytest.raises(ValueError):
        kaggle_like_score(np.ones((1, 70, 35, 4)), np.ones((1, 70, 35, 4)), t, Yr, t_ref)
