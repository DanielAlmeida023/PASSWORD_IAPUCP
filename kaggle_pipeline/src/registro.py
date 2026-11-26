"""Escritura de filas en kaggle_pipeline/experiments/registro.csv (registro único de experimentos)."""

import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from paths import REGISTRO_PATH  # env PASSWORD_OUT_DIR
REGISTRO_COLS = [
    "fecha", "id_exp", "descripcion", "cv_fold1", "cv_fold2", "cv_fold3", "cv_media",
    "skill_RH", "skill_T", "skill_U", "skill_V", "lb_publico", "archivo_submission", "commit",
]


def git_commit_short(repo_dir: Optional[Path] = None) -> str:
    repo_dir = repo_dir or Path(__file__).resolve().parent.parent.parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir, capture_output=True, text=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return ""


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"
    return str(v)


def append_registro(row: Dict, path: Path = REGISTRO_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    full = {c: "" for c in REGISTRO_COLS}
    full["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    full.update({k: _fmt(v) for k, v in row.items() if k in full})
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRO_COLS)
        if new:
            w.writeheader()
        w.writerow(full)
