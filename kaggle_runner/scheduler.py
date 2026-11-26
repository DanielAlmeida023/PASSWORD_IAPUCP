"""
Planificador de Kaggle: mantiene hasta MAX_RUNNING kernels GPU corriendo con una cola de trabajos.
Uso: python kaggle_runner/scheduler.py jobs.txt
jobs.txt: una línea por trabajo -> "<name> | <args de run_cv.py>"  (líneas con # se ignoran)
Al terminar cada kernel baja sus salidas a kaggle_runner/outputs/<name>/ y copia oof/ a
kaggle_pipeline/experiments/oof y refit/ a kaggle_pipeline/experiments/refit.
Estado en kaggle_runner/outputs/scheduler_state.json (reanudable).
"""

import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATE = HERE / "outputs" / "scheduler_state.json"
MAX_RUNNING = 2
POLL_S = 120


def sh(*args: str) -> str:
    r = subprocess.run([sys.executable, str(HERE / "launch.py"), *args], capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def status(name: str) -> str:
    out = sh("status", "--name", name)
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out:
            return s
    return "UNKNOWN"


def collect(name: str) -> None:
    sh("fetch", "--name", name)
    out = HERE / "outputs" / name / "out"
    for sub in ("oof", "refit"):
        src = out / sub
        if src.exists():
            dst = ROOT / "kaggle_pipeline" / "experiments" / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
    # el refit puede escribir directo en out/
    for f in out.glob("*") if out.exists() else []:
        if f.is_file() and f.name.startswith(("model_", "norm_", "pred_test_", "sub_")):
            dst = ROOT / "kaggle_pipeline" / "experiments" / "refit"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)


def main() -> None:
    jobs = []
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, args = [p.strip() for p in line.split("|", 1)]
            jobs.append((name, shlex.split(args)))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    while True:
        running = [n for n, s in st.items() if s in ("RUNNING", "QUEUED", "LAUNCHED")]
        for n in running:
            s = status(n)
            if s in ("COMPLETE", "ERROR", "CANCEL"):
                st[n] = s
                if s == "COMPLETE":
                    collect(n)
                print(f"{time.strftime('%H:%M')} {n} -> {s}", flush=True)
            elif s in ("RUNNING", "QUEUED"):
                st[n] = s
        running = [n for n, s in st.items() if s in ("RUNNING", "QUEUED", "LAUNCHED")]
        pending = [(n, a) for n, a in jobs if n not in st]
        for n, a in pending[: max(0, MAX_RUNNING - len(running))]:
            out = sh("run", "--name", n, "--", *a)
            ok = "successfully pushed" in out.lower()
            st[n] = "LAUNCHED" if ok else "PUSH_FAIL"
            print(f"{time.strftime('%H:%M')} launch {n}: {'ok' if ok else out.strip()[-200:]}", flush=True)
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
        if not pending and not running:
            print("SCHEDULER_DONE", flush=True)
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
