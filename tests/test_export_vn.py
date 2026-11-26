"""
EnsembleVN + export_ensemble.py + paridad producto (Fase 3/4 del plan), sin tocar models/ reales.

1. Dos refits pequeños (run_cv.py --refit --smoke --smoke-real, CPU, width 16, MISMA semilla -> mismas
   stats de normalización) en una carpeta temporal: convnext y fno con --fno-modes 6,4.
2. export_ensemble.export(...) -> modelo.pt / norm_stats.npz / model_card.json + submission en tmp.
   El propio export verifica modelo exportado vs blend de pred_test_*.npy y valida la submission.
3. Producto: app/inti_inference.predict_sequence con INTI_MODELS_DIR=tmp predice las primeras 24 h del
   test y se compara con la submission generada por el export: max|diff| < 1e-3 (incluye redondeo 3 dec.).
Se hace pytest.skip si falta la data real.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
N_HOURS = 24


def test_shared_norm_stats_asserts():
    from ensemble_vn import SHARED_STAT_KEYS, shared_norm_stats

    base = {k: np.ones(3, np.float32) for k in SHARED_STAT_KEYS}
    a = dict(base, lag_set=np.array("default"))
    out = shared_norm_stats([a, dict(a)], ["x_1", "y_2"])
    assert str(out["lag_set"]) == "default"
    with pytest.raises(AssertionError, match="lag_set"):
        shared_norm_stats([a, dict(a, lag_set=np.array("past"))], ["x_1", "y_2"])
    with pytest.raises(AssertionError, match="mu_x"):
        shared_norm_stats([a, dict(a, mu_x=np.full(3, 1.1, np.float32))], ["x_1", "y_2"])


@pytest.fixture(scope="module")
def vn_export(tmp_path_factory):
    for f in ["train.npz", "test.npz", "static.npz", "sample_submission.csv"]:
        if not (DATA_DIR / f).exists():
            pytest.skip(f"falta data/{f}")
    pytest.importorskip("torch")
    import export_ensemble
    import run_cv

    tmp = tmp_path_factory.mktemp("vn")
    common = ["--seed", "7", "--width", "16", "--smoke", "--smoke-real", "--refit", "--no-registro",
              "--out-dir", str(tmp), "--data-dir", str(DATA_DIR)]
    run_cv.main(["--model", "convnext"] + common)
    run_cv.main(["--model", "fno", "--fno-modes", "6,4"] + common)
    blend = tmp / "blend.json"  # mismo formato que blend.py (results/<shrink>/weights), orden distinto a --members
    blend.write_text(json.dumps({"members": ["fno_7", "convnext_7"],
                                 "results": {"0.0": {"weights": [[0.3, 0.6, 0.5, 0.2], [0.7, 0.4, 0.5, 0.8]], "cv_nested": 0.1}}}))
    r = export_ensemble.export(["convnext_7", "fno_7"], str(blend), tmp / "refit", tmp / "models", sub_dir=tmp / "subs",
                               data_dir=DATA_DIR, id_exp="vnsmoke", cv_media=0.1234, lb=None, device="cpu")
    return tmp, r


def test_export_vn_artifacts(vn_export):
    tmp, r = vn_export
    z = np.load(tmp / "models" / "norm_stats.npz")
    assert str(z["arch"]) == "ensemble_vn" and str(z["lag_set"]) == "default"
    cfg = json.loads(str(z["members_config"]))
    assert [c["arch"] for c in cfg] == ["convnext", "fno"] and cfg[1]["fno_modes"] == [6, 4] and cfg[0]["width"] == 16
    np.testing.assert_allclose(z["blend_weights"], [[0.7, 0.4, 0.5, 0.8], [0.3, 0.6, 0.5, 0.2]], atol=1e-6)
    card = json.loads((tmp / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert card["arch"] == "ensemble_vn" and card["validation"]["cv_media"] == 0.1234 and len(card["members"]) == 2
    assert card["weights"]["sha256"] == r["sha256"] and r["submission"].name.endswith("_vnsmoke_cv0.1234.csv")
    assert max(r["diff"].values()) < 0.05  # modelo exportado vs blend de pred_test (float16)


def test_paridad_producto_vs_export_vn(vn_export, monkeypatch):
    import pandas as pd

    tmp, r = vn_export
    sys.path.insert(0, str(ROOT / "app"))
    import inti_inference as ii

    monkeypatch.setenv("INTI_MODELS_DIR", str(tmp / "models"))
    ii.reset_runner()
    try:
        te = np.load(DATA_DIR / "test.npz")
        Xte, tte = te["X"], te["times"].astype(str)
        static = dict(np.load(DATA_DIR / "static.npz"))
        assert ii.configured_lag_set() == "default"
        Y = ii.predict_sequence(Xte, tte, static, out_idx=np.arange(N_HOURS))
        assert ii.get_runner(static)["arch"] == "ensemble_vn"
    finally:
        ii.reset_runner()  # no dejar cargado el modelo temporal para otros tests

    sub = pd.read_csv(r["submission"])
    parts = sub["id"].str.split("_", expand=True)
    pos = {t: i for i, t in enumerate(tte[:N_HOURS])}
    ti = parts[0].map(pos)
    m = ti.notna().values
    assert m.sum() == 8 * 70 * 35
    ti, rr, cc = ti[m].astype(int).values, parts[1][m].astype(int).values, parts[2][m].astype(int).values
    diffs = {v: float(np.abs(Y[ti, rr, cc, k] - sub.loc[m, v].values).max()) for k, v in enumerate(["RH", "T", "U", "V"])}
    print(f"max|diff| producto vs submission del export: {diffs}")
    assert max(diffs.values()) < 1e-3, diffs


def test_export_vn_fp16(vn_export):
    """Con umbral bajo se guarda en float16; el producto (carga con .float()) sigue igual a esa submission."""
    import export_ensemble

    tmp, _ = vn_export
    r16 = export_ensemble.export(["convnext_7", "fno_7"], "uniform", tmp / "refit", tmp / "models16", sub_dir=tmp / "subs16",
                                 data_dir=DATA_DIR, id_exp="vn16", device="cpu", fp16_threshold_mb=0.5, tol=0.2)
    import torch

    sd = torch.load(tmp / "models16" / "modelo.pt", map_location="cpu", weights_only=True)
    assert r16["dtype"] == "float16" and any(v.dtype == torch.float16 for v in sd.values())
    assert sd["blend_weights"].dtype == torch.float32  # buffers en float32
