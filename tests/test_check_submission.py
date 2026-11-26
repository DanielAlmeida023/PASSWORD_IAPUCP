import numpy as np
import pandas as pd
import pytest

from check_submission import EXPECTED_ROWS, build_submission, check_submission_df


def _sample(n_t=2):
    times = [f"2024070{2 + i}00" for i in range(n_t)]
    ids = [f"{t}_{r:02d}_{c:02d}" for t in times for r in range(70) for c in range(35)]
    df = pd.DataFrame({"id": ids, "RH": 50.0, "T": 280.0, "U": 0.0, "V": 0.0})
    return df, np.array(times)


def test_expected_rows_constant():
    # sample_submission solo trae las horas evaluadas (00,03,...,21): 792 h / 3 = 264 h
    assert EXPECTED_ROWS == 264 * 70 * 35 == 646800


def test_build_and_validate_ok():
    sample, times = _sample()
    rng = np.random.default_rng(0)
    preds = np.stack([rng.uniform(10, 90, (2, 70, 35)), rng.uniform(270, 290, (2, 70, 35)),
                      rng.normal(0, 2, (2, 70, 35)), rng.normal(0, 2, (2, 70, 35))], axis=-1)
    sub = build_submission(preds, times, sample)
    assert list(sub.columns) == ["id", "RH", "T", "U", "V"]
    assert sub.loc[70 * 35 + 35 + 1, "T"] == pytest.approx(round(preds[1, 1, 1, 1], 3))
    errors, warns = check_submission_df(sub, sample, expected_rows=len(sample))
    assert errors == [] and warns == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.drop(index=0), "Filas"),
    (lambda s: s[["id", "T", "RH", "U", "V"]], "Columnas"),
    (lambda s: s.assign(RH=np.where(np.arange(len(s)) == 3, np.nan, s["RH"])), "NaN"),
    (lambda s: s.assign(U=np.where(np.arange(len(s)) == 3, np.inf, s["U"])), "inf"),
    (lambda s: s.assign(id=s["id"].iloc[::-1].values), "ids"),
    (lambda s: s.assign(V=0.0), "constante"),
])
def test_detects_errors(mutate, needle):
    sample, times = _sample()
    rng = np.random.default_rng(1)
    sub = sample.copy()
    for col in ["RH", "T", "U", "V"]:
        sub[col] = sub[col] + rng.normal(0, 1, len(sub))
    errors, _ = check_submission_df(mutate(sub), sample, expected_rows=len(sample))
    assert any(needle in e for e in errors), errors
