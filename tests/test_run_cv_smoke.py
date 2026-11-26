"""Smoke de run_cv.py (data sintética, 1 época, CPU): CV de 1 fold y refit no crashean y escriben salidas."""
import csv

import numpy as np
import pytest

pytest.importorskip("torch")


def test_run_cv_smoke_fold3(tmp_path):
    import run_cv
    from oof_io import load_oof

    s = run_cv.main(["--model", "fno", "--smoke", "--folds", "3", "--fno-modes", "6,4", "--loss-w-uv", "1.3",
                     "--recency-weights", "plan", "--out-dir", str(tmp_path)])
    assert np.isfinite(s["cv_media"])
    pred, t = load_oof("fno", 304, 3, tmp_path / "oof")
    assert pred.shape[1:] == (70, 35, 4) and len(t) == len(pred) and min(t) >= "2024061600"
    rows = list(csv.DictReader(open(tmp_path / "registro.csv", encoding="utf-8")))
    assert rows[-1]["id_exp"] == "cv_fno_s304_SMOKE" and rows[-1]["cv_fold3"] and not rows[-1]["cv_fold1"]


def test_run_cv_smoke_refit(tmp_path):
    import run_cv
    from oof_io import load_pred_test

    r = run_cv.main(["--model", "fno", "--smoke", "--refit", "--out-dir", str(tmp_path)])
    pred, t = load_pred_test("fno", 304, tmp_path / "refit")
    assert pred.shape[1:] == (70, 35, 4) and np.isfinite(pred).all() and t[0] == "2024070200"
    assert (tmp_path / "refit" / "model_fno_304.pt").exists() and (tmp_path / "refit" / "norm_fno_304.npz").exists()
