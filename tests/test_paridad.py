"""
Test de paridad producto <-> submission (Fase 4 del plan; regla: el producto debe usar los MISMOS pesos).

Carga models/modelo.pt + models/norm_stats.npz con el MISMO código de inferencia del producto
(app/inti_inference.predict_sequence, que importa product_app/backend/app/main.py y
app/ejemplo_producto.predecir), predice las primeras 24 h del test y compara contra las mismas filas
de submission.csv: max |diff| < 1e-3 de modelo + 5e-4 de redondeo (submission con 3 decimales).
Medido (CPU vs salida GPU del notebook): ~6e-4 en RH -> 1.1e-3 contando el redondeo.

Las features se construyen sobre la secuencia de test COMPLETA (igual que el notebook) y luego se
recortan a las 24 primeras horas, para que los lags ±24 h y M24/S24 sean idénticos.
Se hace pytest.skip si falta data, pesos o submission.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"
SUB_CANDIDATES = [ROOT / "submission.csv", ROOT / "train" / "submission.csv"]
TOL = 1e-3  # diferencia permitida del MODELO (CPU vs GPU, orden de ops)
ROUND = 5e-4  # submission.csv está redondeada a 3 decimales
N_HOURS = 24


def _need(path: Path):
    if not path.exists():
        pytest.skip(f"falta {path.relative_to(ROOT)}")


# Primeras 5 filas de submission.csv tal como quedaron impresas (sub.head()) en la salida guardada de
# train/submission_notebook.ipynb (envío 0.49844). Permite un chequeo de paridad aun sin submission.csv.
NOTEBOOK_HEAD = {
    "2024070200_00_00": (96.921, 289.054, -0.367, 4.734),
    "2024070200_00_01": (95.430, 288.922, 0.077, 4.030),
    "2024070200_00_02": (91.753, 289.019, 0.483, 3.603),
    "2024070200_00_03": (87.260, 289.162, 0.478, 3.044),
    "2024070200_00_04": (83.139, 289.446, 0.305, 2.367),
}


def _load_product():
    for p in [DATA_DIR / "test.npz", DATA_DIR / "static.npz", ROOT / "models" / "modelo.pt", ROOT / "models" / "norm_stats.npz"]:
        _need(p)
    pytest.importorskip("torch")
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    try:
        from inti_inference import predict_sequence  # código del producto
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"no se pudo importar app/inti_inference.py: {e}")
    te = np.load(DATA_DIR / "test.npz")
    return predict_sequence, te["X"], te["times"].astype(str), dict(np.load(DATA_DIR / "static.npz"))


def test_paridad_producto_vs_head_notebook():
    predict_sequence, Xte, tte, static = _load_product()
    assert tte[0] == "2024070200"
    Y = predict_sequence(Xte, tte, static, out_idx=np.arange(1))
    got = {f"2024070200_00_{c:02d}": Y[0, 0, c] for c in range(5)}
    diff = max(float(np.abs(got[k] - np.array(v)).max()) for k, v in NOTEBOOK_HEAD.items())
    print(f"max|diff| producto vs head del notebook: {diff:.2e}")
    assert diff < TOL + ROUND, f"PARIDAD FALLIDA vs salida del notebook: {diff}"


def test_paridad_producto_vs_submission():
    for p in [DATA_DIR / "test.npz", DATA_DIR / "static.npz", ROOT / "models" / "modelo.pt", ROOT / "models" / "norm_stats.npz"]:
        _need(p)
    sub_path = next((p for p in SUB_CANDIDATES if p.exists()), None)
    if sub_path is None:
        pytest.skip("falta submission.csv (generarlo con train/submission_notebook.ipynb)")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("torch")
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    try:
        from inti_inference import predict_sequence  # código del producto
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"no se pudo importar app/inti_inference.py: {e}")

    te = np.load(DATA_DIR / "test.npz")
    Xte, tte = te["X"], te["times"].astype(str)
    static = dict(np.load(DATA_DIR / "static.npz"))

    Y = predict_sequence(Xte, tte, static, out_idx=np.arange(N_HOURS))
    assert Y.shape == (N_HOURS, 70, 35, 4)

    sub = pd.read_csv(sub_path)
    parts = sub["id"].str.split("_", expand=True)
    pos = {t: i for i, t in enumerate(tte[:N_HOURS])}
    ti = parts[0].map(pos)
    m = ti.notna().values
    n_kaggle = sum(int(t[-2:]) % 3 == 0 for t in tte[:N_HOURS])  # submission solo trae horas 00,03,..,21
    assert m.sum() == n_kaggle * 70 * 35, f"filas de submission para las 24 h: {m.sum()}"
    ti = ti[m].astype(int).values
    r = parts[1][m].astype(int).values
    c = parts[2][m].astype(int).values

    diffs = {}
    for k, v in enumerate(["RH", "T", "U", "V"]):
        diffs[v] = float(np.abs(Y[ti, r, c, k] - sub.loc[m, v].values).max())
    print(f"max|diff| por variable (producto vs {sub_path.name}): {diffs}")
    assert max(diffs.values()) < TOL + ROUND, f"PARIDAD FALLIDA: {diffs} (el producto no usa el mismo modelo/preprocesamiento)"
