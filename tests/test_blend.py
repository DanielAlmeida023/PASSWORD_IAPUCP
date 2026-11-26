import numpy as np
import pytest

from blend import (
    blend_predict,
    evaluate_blend,
    fit_blend,
    make_fold_oof,
    nested_blend_cv,
    project_simplex,
    solve_nnls_sum1,
    solve_simplex_qp,
)
from cv import fold_masks
from metrics import clim_by_hour
from oof_io import load_oof, save_oof
from synthetic_data import hourly_times, make_dataset

SIGMAS = (0.5, 1.0, 2.0)


@pytest.fixture(scope="module")
def fold_oofs():
    t = hourly_times(step_h=3)
    _, Y = make_dataset(t, seed=5)
    rng = np.random.default_rng(7)
    out = []
    for k in (1, 2, 3):
        trn, val = fold_masks(t, k)
        yv = Y[val]
        preds = [yv + rng.normal(0, s, yv.shape).astype(np.float32) for s in SIGMAS]
        preds.append(yv + 5.0)  # miembro malo: sesgo grande
        out.append(make_fold_oof(k, np.stack(preds), yv, t[val], clim_by_hour(Y[trn], t[trn])))
    return out


def test_project_simplex():
    for v in [np.array([0.2, 0.3, 0.5]), np.array([3.0, -1.0, 0.0]), np.random.default_rng(0).normal(size=7)]:
        w = project_simplex(v)
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(project_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])


def test_simplex_qp_matches_nnls():
    rng = np.random.default_rng(1)
    for _ in range(5):
        A = rng.normal(size=(200, 4))
        y = A @ np.array([0.5, 0.3, 0.2, -0.1]) + 0.1 * rng.normal(size=200)
        G, b = A.T @ A / 200, A.T @ y / 200
        w1, w2 = solve_simplex_qp(G, b), solve_nnls_sum1(G, b)
        assert w1.sum() == pytest.approx(1) and w1.min() >= 0
        np.testing.assert_allclose(w1, w2, atol=2e-3)


def test_inverse_variance_weights(fold_oofs):
    W = fit_blend(fold_oofs)
    assert W.shape == (4, 4)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)
    assert (W >= 0).all()
    inv = np.array([1 / s**2 for s in SIGMAS])
    expected = np.append(inv / inv.sum(), 0.0)
    for v in range(4):
        np.testing.assert_allclose(W[:, v], expected, atol=0.02)
    W_nnls = fit_blend(fold_oofs, method="nnls")
    np.testing.assert_allclose(W, W_nnls, atol=3e-3)


def test_shrinkage(fold_oofs):
    W0 = fit_blend(fold_oofs, shrink=0.0)
    W3 = fit_blend(fold_oofs, shrink=0.3)
    np.testing.assert_allclose(W3, 0.7 * W0 + 0.3 / 4, atol=1e-6)
    np.testing.assert_allclose(fit_blend(fold_oofs, shrink=1.0), 0.25, atol=1e-6)


def test_blend_beats_members_and_nested(fold_oofs):
    W = fit_blend(fold_oofs)
    r = evaluate_blend(fold_oofs, W)
    best_single = evaluate_blend(fold_oofs, np.eye(4, dtype=np.float32)[:, [0]].repeat(4, axis=1))
    assert r["cv_media"] > best_single["cv_media"]
    assert set(r["folds"]) == {1, 2, 3}
    nest = nested_blend_cv(fold_oofs)
    assert set(nest["weights"]) == {1, 2, 3}
    assert abs(nest["cv_media"] - r["cv_media"]) < 0.01  # mismo proceso en todos los folds


def test_blend_predict_shapes(fold_oofs):
    f = fold_oofs[0]
    out = blend_predict(f.preds, np.full((4, 4), 0.25, np.float32))
    np.testing.assert_allclose(out, f.preds.mean(0), atol=1e-4)


def test_oof_float16_roundtrip(tmp_path):
    t = hourly_times("2024061600", "2024063023")
    _, Y = make_dataset(t, seed=2)
    save_oof(Y, t, "unet", 203, 3, tmp_path)
    assert np.load(tmp_path / "oof_unet_203_fold3.npy").dtype == np.float16
    Y2, t2 = load_oof("unet", 203, 3, tmp_path)
    assert (t2 == t).all()
    err = np.abs(Y2 - Y).reshape(-1, 4).max(axis=0)
    assert err[1] < 0.02  # T guardada en °C: paso float16 <= 0.016 K
    assert err[0] < 0.04 and err[2] < 0.01 and err[3] < 0.01
