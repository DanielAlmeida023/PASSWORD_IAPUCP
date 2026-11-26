"""
Envíos de calibración (Fase 1.4) con el EnsembleV8 ya entrenado (models/modelo.pt):
  1. baseline  : blend original (debe reproducir el envío 0.49844)
  2. sin_fno   : pesos originales renormalizados sin el miembro FNO
  3. uniforme  : 0.25 cada miembro
Predice cada miembro una sola vez con el mismo código del producto (app/inti_inference.py).
Uso: python kaggle_runner/make_calibration_subs.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "kaggle_pipeline" / "src"))

import torch  # noqa: E402
import inti_inference as ii  # noqa: E402
from check_submission import build_submission, check_submission_df  # noqa: E402
from train import extract_hour_utc  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)


def main() -> None:
    te = np.load(DATA / "test.npz")
    static = dict(np.load(DATA / "static.npz"))
    X, t = te["X"], te["times"].astype(str)
    runner = ii.get_runner(static)
    model, dev = runner["model"], runner["device"]
    W, M24, S24, GRAD, TD = ii.build_features(X, t, lag_set=runner["lag_set"])
    f = ii.normalize_features(W, M24, S24, GRAD, TD, runner["stats"])
    h = extract_hour_utc(t)
    members = [model.m_conv, model.m_ms, model.m_unet, model.m_fno]
    preds = np.zeros((4, len(t), 70, 35, 4), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(t), 16):
            sl = slice(s, s + 16)
            args = [torch.tensor(f[k][sl], device=dev) for k in ("Wn", "Mn", "Sn", "Gn", "Tn")]
            args.append(torch.tensor(h[sl], dtype=torch.long, device=dev))
            for k, m in enumerate(members):
                preds[k, sl] = m(*args).cpu().numpy()
    print(f"miembros predichos en {time.time()-t0:.0f}s ({dev})")
    np.save(ROOT / "local_runs" / "v8_members_test.npy", preds.astype(np.float16))

    bw = model.blend_weights.cpu().numpy()  # (4 miembros, 4 vars)
    no_fno = bw.copy()
    no_fno[3] = 0.0
    no_fno /= no_fno.sum(axis=0, keepdims=True)
    configs = {
        "v8_baseline": bw,
        "v8_sin_fno": no_fno,
        "v8_uniforme": np.full((4, 4), 0.25, dtype=np.float32),
    }
    sample = pd.read_csv(DATA / "sample_submission.csv")
    fecha = time.strftime("%Y%m%d")
    for name, w in configs.items():
        y = np.einsum("mv,mnhwv->nhwv", w, preds)
        y[..., 0] = np.clip(y[..., 0], 0.0, 100.0)
        sub = build_submission(y, t, sample)
        check_submission_df(sub, sample)
        path = OUT / f"sub_{fecha}_{name}_cvNA.csv"
        sub.to_csv(path, index=False)
        print(f"{name}: pesos (filas=Conv,MS,UNet,FNO; cols=RH,T,U,V)\n{np.round(w, 3)}\n -> {path.name}")


if __name__ == "__main__":
    main()
