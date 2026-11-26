"""
Opciones de viento de run_cv.py (Fase 2.5): --topo-extra y --wind-head, y post-proceso por hora (2.7).

1. Default IDÉNTICO: build_member_model sin opciones = misma arquitectura/init/salida que antes del cambio
   (nº de parámetros y claves fijos; hash del state_dict y suma de la salida con la versión de torch en la
   que se tomó la referencia).
2. Parser: flags apagados por defecto, nombre de miembro sin sufijo.
3. topo_extra: forma, normalización, eje del valle NNO–SSE en el DEM oficial.
4. Miembro con opciones: forward, state_dict distinto, reconstrucción vía members_config (EnsembleVN).
5. Smoke CV sintético y refit real -> export -> producto (app/inti_inference) con un miembro -topox-wh.
6. postprocess: recupera un sesgo/escala conocidos y default = identidad.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Referencia tomada ANTES de añadir las opciones (w=16, torch.manual_seed(123), make_static(seed=0)).
REF_TORCH = "2.11.0+cpu"
REF = {
    "convnext": dict(n=349253, keys=126, sha="30249338d646057a43d4ee52aa54a3b412d894fbb61a687f9a3200c9362fa98d", ysum=1688478.053860494),
    "multiscale": dict(n=352773, keys=145, sha="2d724f4714754aab6d14827ccf7e66e79ead740ccf533254d084658060f84a0e", ysum=1686139.8727983716),
    "unet": dict(n=459037, keys=134, sha="ae01acd3514a9dbd55c98286b4dc7ea8c209fd330cbc8bb68191c6780873106d", ysum=1691063.021207239),
    "fno": dict(n=578213, keys=92, sha="538f7f9db1cab54bd58376ba2c45b3cdc600801844b6da3727bbb93193e58548", ysum=1685368.6191894896),
}


def _ctx():
    from ensemble_vn import build_context
    from synthetic_data import make_static

    st = make_static(seed=0)
    stats = {"mu_x": np.array([60, 285, 1, -1], np.float32), "sd_x": np.array([20, 5, 3, 3], np.float32),
             "anom_sd": np.ones((70, 35, 4), np.float32)}
    return build_context(st, stats)


def _inputs():
    torch.manual_seed(0)
    return (torch.randn(2, 9, 17, 13, 4), torch.randn(2, 17, 13, 4), torch.rand(2, 17, 13, 4),
            torch.randn(2, 6, 17, 13), torch.randn(2, 1, 17, 13), torch.tensor([3, 15]))


def _build(arch, ctx, **kw):
    from ensemble_vn import build_member_model

    torch.manual_seed(123)
    return build_member_model(arch, ctx["grid"], ctx["topo"], ctx["lapse_delta_norm"], ctx["dem_delta_km"], ctx["mu_x"],
                              ctx["sd_x"], ctx["anom_sd"], w=16, dropout=0.0, **kw).eval()


@pytest.mark.parametrize("arch", sorted(REF))
def test_default_model_identical(arch):
    ctx, inp = _ctx(), _inputs()
    m = _build(arch, ctx)
    sd = m.state_dict()
    assert sum(p.numel() for p in m.parameters()) == REF[arch]["n"] and len(sd) == REF[arch]["keys"]
    assert m.n_topo == 10 and not m.wind_head
    m2 = _build(arch, ctx, topo_extra=None, wind_head=False)  # flags explícitos apagados = default
    with torch.no_grad():
        y, y2 = m(*inp), m2(*inp)
    assert torch.equal(y, y2)
    if torch.__version__ == REF_TORCH:
        h = hashlib.sha256()
        for k in sd:
            h.update(k.encode())
            h.update(sd[k].detach().numpy().tobytes())
        assert h.hexdigest() == REF[arch]["sha"]
        assert abs(float(y.double().sum()) - REF[arch]["ysum"]) < 1e-6 * abs(REF[arch]["ysum"])


def test_parser_defaults_unchanged():
    import run_cv

    a = run_cv.build_parser().parse_args(["--model", "unet", "--seed", "42"])
    assert a.topo_extra is False and a.wind_head is False
    hp = run_cv._resolve_hparams(a)
    assert hp["member"] == "unet" and hp["topo_extra"] is False and hp["wind_head"] is False
    a2 = run_cv.build_parser().parse_args(["--model", "unet", "--topo-extra", "--wind-head"])
    assert run_cv._resolve_hparams(a2)["member"] == "unet-topox-wh"


def test_topo_extra_features_synthetic():
    from synthetic_data import make_static
    from topo_extra import N_TOPO_EXTRA, build_topo_extra_features

    st = make_static(seed=0)
    f, info = build_topo_extra_features(st["dem"], st["wrf_lats"], st["wrf_lons"])
    assert f.shape == (N_TOPO_EXTRA, 70, 35) and f.dtype == np.float32 and np.isfinite(f).all()
    z = [0, 1, 2, 5, 6, 7, 8]  # canales z-score
    np.testing.assert_allclose(f[z].mean(axis=(1, 2)), 0, atol=1e-4)
    assert np.all(np.abs(f[3:5]) <= 1.0 + 1e-6) and 0 <= info["bearing_deg"] < 360


def test_valley_axis_official_dem():
    p = DATA_DIR / "static.npz"
    if not p.exists():
        pytest.skip("falta data/static.npz")
    from topo_extra import estimate_valley_axis

    st = np.load(p)
    info = estimate_valley_axis(st["dem"], st["wrf_lats"], st["wrf_lons"])
    # Callejón de Huaylas: NNO–SSE (rumbo del eje hacia el norte entre 300° y 355°), valle alargado
    assert 300 <= info["bearing_deg"] <= 355 and info["elongation"] > 3, info


@pytest.mark.parametrize("arch", ["convnext", "multiscale", "unet", "fno"])
def test_member_with_options_roundtrip(arch):
    from ensemble_vn import EnsembleVN

    ctx, inp = _ctx(), _inputs()
    m = _build(arch, ctx, topo_extra=ctx["topo_extra"], wind_head=True,
               **({"fno_modes": (6, 4)} if arch == "fno" else {}))
    assert m.n_topo == 10 + ctx["topo_extra"].shape[0] and m.wind_head
    m0 = _build(arch, ctx, **({"fno_modes": (6, 4)} if arch == "fno" else {}))
    assert sum(p.numel() for p in m.parameters()) > sum(p.numel() for p in m0.parameters())
    with torch.no_grad():
        y = m(*inp)
    assert y.shape == (2, 70, 35, 4) and torch.isfinite(y).all()
    cfg = [{"name": f"{arch}-topox-wh_1", "arch": arch, "width": 16, "fno_modes": [6, 4] if arch == "fno" else None,
            "topo_extra": True, "wind_head": True}]
    ens = EnsembleVN(ctx["grid"], ctx["topo"], ctx["lapse_delta_norm"], ctx["dem_delta_km"], ctx["mu_x"], ctx["sd_x"],
                     ctx["anom_sd"], cfg, np.ones((1, 4), np.float32), topo_extra=ctx["topo_extra"]).eval()
    ens.load_members([m.state_dict()])  # estricto: formas compatibles
    with torch.no_grad():
        ye = ens(*inp)
    y_c = torch.cat([y[..., :1].clamp(0, 100), y[..., 1:]], -1)
    assert torch.allclose(ye, y_c, atol=1e-5)
    # un miembro default no puede cargar el state_dict con opciones
    with pytest.raises(RuntimeError):
        EnsembleVN(ctx["grid"], ctx["topo"], ctx["lapse_delta_norm"], ctx["dem_delta_km"], ctx["mu_x"], ctx["sd_x"],
                   ctx["anom_sd"], [dict(cfg[0], topo_extra=False, wind_head=False)], np.ones((1, 4), np.float32),
                   topo_extra=ctx["topo_extra"]).load_members([m.state_dict()])


def test_member_config_from_norm_options():
    from ensemble_vn import member_config_from_norm

    base = {"arch": np.array("unet"), "width": np.array(64), "fno_modes": np.array([12, 8])}
    c = member_config_from_norm(base, "unet_42")
    assert c["topo_extra"] is False and c["wind_head"] is False
    c = member_config_from_norm(dict(base, topo_extra=np.array(True), wind_head=np.array(True)), "unet-topox-wh_42")
    assert c["topo_extra"] is True and c["wind_head"] is True and c["member"] == "unet-topox-wh"


def test_run_cv_smoke_options_synthetic(tmp_path):
    import run_cv
    from oof_io import load_oof

    s = run_cv.main(["--model", "unet", "--width", "16", "--smoke", "--folds", "3", "--topo-extra", "--wind-head",
                     "--out-dir", str(tmp_path), "--no-registro"])
    assert np.isfinite(s["cv_media"]) and s["member"] == "unet-topox-wh"
    assert s["hparams"]["topo_extra"] is True and s["hparams"]["wind_head"] is True
    pred, _ = load_oof("unet-topox-wh", 203, 3, tmp_path / "oof")
    assert pred.shape[1:] == (70, 35, 4) and np.isfinite(pred).all()


def test_refit_export_producto_options(tmp_path, monkeypatch):
    """Refit real (smoke) con --topo-extra --wind-head -> export_ensemble -> inti_inference: misma salida."""
    for f in ["train.npz", "test.npz", "static.npz", "sample_submission.csv"]:
        if not (DATA_DIR / f).exists():
            pytest.skip(f"falta data/{f}")
    import export_ensemble
    import run_cv

    run_cv.main(["--model", "convnext", "--seed", "7", "--width", "16", "--smoke", "--smoke-real", "--refit",
                 "--topo-extra", "--wind-head", "--no-registro", "--out-dir", str(tmp_path), "--data-dir", str(DATA_DIR)])
    z = np.load(tmp_path / "refit" / "norm_convnext-topox-wh_7.npz")
    assert bool(z["topo_extra"]) and bool(z["wind_head"]) and 300 <= float(z["valley_axis_deg"]) <= 355
    r = export_ensemble.export(["convnext-topox-wh_7"], "uniform", tmp_path / "refit", tmp_path / "models",
                               sub_dir=tmp_path / "subs", data_dir=DATA_DIR, id_exp="wsmoke", device="cpu")
    assert max(r["diff"].values()) < 0.05
    cfg = json.loads(str(np.load(tmp_path / "models" / "norm_stats.npz")["members_config"]))
    assert cfg[0]["topo_extra"] is True and cfg[0]["wind_head"] is True

    sys.path.insert(0, str(ROOT / "app"))
    import inti_inference as ii
    from oof_io import load_pred_test

    monkeypatch.setenv("INTI_MODELS_DIR", str(tmp_path / "models"))
    ii.reset_runner()
    try:
        te = np.load(DATA_DIR / "test.npz")
        Xte, tte = te["X"], te["times"].astype(str)
        static = dict(np.load(DATA_DIR / "static.npz"))
        Y = ii.predict_sequence(Xte, tte, static, out_idx=np.arange(12))
    finally:
        ii.reset_runner()
    pred, _ = load_pred_test("convnext-topox-wh", 7, tmp_path / "refit")
    diff = np.abs(Y - pred[:12]).max(axis=(0, 1, 2))
    assert diff.max() < 0.05, diff  # float16 del pred_test


def test_postprocess_fit_apply():
    from postprocess import apply_hourly_affine, fit_hourly_affine, identity_params, params_from_stats

    rng = np.random.default_rng(0)
    t = np.array([f"20240501{h:02d}" for h in range(24)] * 5)
    y = rng.normal(size=(len(t), 4, 3, 4)).astype(np.float32) + 50
    p = y.copy()
    p[..., 2] -= 0.5  # sesgo conocido en U
    pb = fit_hourly_affine(p, y, t, mode="bias")
    np.testing.assert_allclose(pb["pp_b"][:, 2], 0.5, atol=1e-5)
    np.testing.assert_allclose(apply_hourly_affine(p, t, pb), y, atol=1e-4)
    np.testing.assert_allclose(apply_hourly_affine(y, t, identity_params()), y)
    p2 = 50 + 0.5 * (y - 50)  # amplitud reducida a la mitad
    pa = fit_hourly_affine(p2, y, t, mode="affine", vars_idx=(3,))
    np.testing.assert_allclose(pa["pp_a"][:, 3], 2.0, atol=1e-3)
    assert np.all(pa["pp_a"][:, :3] == 1) and params_from_stats({}) is None
    assert apply_hourly_affine(p, t, None) is p
