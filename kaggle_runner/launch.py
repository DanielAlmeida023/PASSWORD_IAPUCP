"""
Lanzador de experimentos en Kaggle GPU (notebooks PRIVADOS) — Equipo PASSWORD.

Flujo:
  1. `python kaggle_runner/launch.py sync`  -> sube kaggle_pipeline/src como dataset PRIVADO
     `<user>/password-src` (crea o versiona).
  2. `python kaggle_runner/launch.py run --name cv-unet-s42 -- --model unet --seed 42 --folds 1,2,3`
     -> crea/actualiza un kernel PRIVADO con GPU que ejecuta `run_cv.py <args>` sobre la data oficial.
  3. `python kaggle_runner/launch.py status --name cv-unet-s42`
  4. `python kaggle_runner/launch.py fetch --name cv-unet-s42` -> baja salidas a kaggle_runner/outputs/<name>/

Solo usa la data oficial de la competencia (competition_sources). Nada se publica: is_private=true.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "kaggle_pipeline" / "src"
OUT = Path(__file__).resolve().parent / "outputs"
COMPETITION = "hackathon-ia-pucp-2026"
DATASET_SLUG = "password-src"


def kaggle_user() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(cfg.read_text(encoding="utf-8"))["username"]


def kg(*args: str, check: bool = True) -> str:
    cmd = [sys.executable, "-m", "kaggle", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    if check and res.returncode != 0:
        raise SystemExit(f"[kaggle] fallo: {' '.join(args)}\n{out}")
    return out


def cmd_sync(_: argparse.Namespace) -> None:
    user = kaggle_user()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ds"
        shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        meta = {"title": "password-src", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
        (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        status = kg("datasets", "status", f"{user}/{DATASET_SLUG}", check=False)
        if "ready" in status.lower():
            print(kg("datasets", "version", "-p", str(d), "-m", f"sync {time.strftime('%Y-%m-%d %H:%M')}", "-r", "zip"))
        else:
            print(kg("datasets", "create", "-p", str(d), "-r", "zip"))  # privado por defecto
    print("[sync] OK. Espera ~1 min a que Kaggle procese el dataset antes de lanzar kernels.")


KERNEL_TEMPLATE = r'''
import os, sys, shutil, subprocess, glob
src_candidates = glob.glob("/kaggle/input/**/run_cv.py", recursive=True)
assert src_candidates, "No se encontró run_cv.py en el dataset password-src"
src_dir = os.path.dirname(src_candidates[0])
work_src = "/kaggle/working/src"
shutil.copytree(src_dir, work_src, dirs_exist_ok=True)
data_candidates = glob.glob("/kaggle/input/**/train.npz", recursive=True)
assert data_candidates, "No se encontró train.npz"
os.environ["PASSWORD_DATA_DIR"] = os.path.dirname(data_candidates[0])
os.environ["PASSWORD_OUT_DIR"] = "/kaggle/working/out"
os.makedirs("/kaggle/working/out", exist_ok=True)
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    shutil.rmtree(work_src, ignore_errors=True)
    raise SystemExit("SIN GPU: abortado para no gastar cuota (revisa verificación telefónica / acelerador)")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
args = __ARGS__
print("CMD:", args, flush=True)
r = subprocess.run([sys.executable, "-u", os.path.join(work_src, __SCRIPT__)] + args, cwd=work_src)
shutil.rmtree(work_src, ignore_errors=True)
sys.exit(r.returncode)
'''


def cmd_run(ns: argparse.Namespace) -> None:
    user = kaggle_user()
    slug = f"password-{ns.name}".lower()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        code = KERNEL_TEMPLATE.replace("__ARGS__", repr(ns.args)).replace("__SCRIPT__", repr(ns.script))
        (d / "run.py").write_text(code, encoding="utf-8")
        meta = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": False,
            "dataset_sources": [f"{user}/{DATASET_SLUG}"],
            "competition_sources": [COMPETITION],
            "kernel_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(kg("kernels", "push", "-p", str(d)))
    print(f"[run] lanzado {user}/{slug}")


def cmd_status(ns: argparse.Namespace) -> None:
    print(kg("kernels", "status", f"{kaggle_user()}/password-{ns.name}".lower(), check=False))


def cmd_fetch(ns: argparse.Namespace) -> None:
    dest = OUT / ns.name
    dest.mkdir(parents=True, exist_ok=True)
    print(kg("kernels", "output", f"{kaggle_user()}/password-{ns.name}".lower(), "-p", str(dest), "--force", check=False))
    print(f"[fetch] salidas en {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--script", default="run_cv.py")
    r.add_argument("args", nargs=argparse.REMAINDER)
    for c in ("status", "fetch"):
        s = sub.add_parser(c)
        s.add_argument("--name", required=True)
    ns = p.parse_args()
    if getattr(ns, "args", None) and ns.args and ns.args[0] == "--":
        ns.args = ns.args[1:]
    {"sync": cmd_sync, "run": cmd_run, "status": cmd_status, "fetch": cmd_fetch}[ns.cmd](ns)


if __name__ == "__main__":
    main()
