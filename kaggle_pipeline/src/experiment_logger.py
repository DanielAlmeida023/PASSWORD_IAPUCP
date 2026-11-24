"""
Registro ligero y automático de experimentos locales en el repositorio.
Guarda métricas de Validación Cruzada (CV), hiperparámetros, predicciones OOF y submissions
para comparar modelos del equipo antes de gastar intentos diarios en Kaggle.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
LOG_CSV_PATH = EXPERIMENTS_DIR / "experiments_log.csv"

CSV_HEADERS = [
    "exp_id",
    "timestamp",
    "author",
    "model_name",
    "cv_strategy",
    "metric_name",
    "cv_mean",
    "cv_std",
    "seed",
    "n_features",
    "notes",
    "submitted_to_kaggle",
    "public_lb_score",
]


class ExperimentLogger:
    def __init__(self, experiments_dir: Path = EXPERIMENTS_DIR):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.experiments_dir / "experiments_log.csv"
        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def log_experiment(
        self,
        model_name: str,
        metric_name: str,
        fold_scores: List[float],
        params: Dict[str, Any],
        features: List[str],
        oof_preds: Optional[np.ndarray] = None,
        test_preds: Optional[np.ndarray] = None,
        author: str = "team",
        cv_strategy: str = "5fold_stratified",
        seed: int = 42,
        notes: str = "",
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}_{model_name}"
        exp_folder = self.experiments_dir / exp_id
        exp_folder.mkdir(parents=True, exist_ok=True)

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))

        # Guardar configuración detallada del experimento
        config_data = {
            "exp_id": exp_id,
            "timestamp": timestamp,
            "author": author,
            "model_name": model_name,
            "cv_strategy": cv_strategy,
            "metric_name": metric_name,
            "fold_scores": fold_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "seed": seed,
            "features": features,
            "params": params,
            "notes": notes,
        }
        with open(exp_folder / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        if oof_preds is not None:
            np.save(exp_folder / "oof_preds.npy", oof_preds)
        if test_preds is not None:
            np.save(exp_folder / "test_preds.npy", test_preds)

        # Añadir fila a experiments_log.csv
        with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                exp_id,
                timestamp,
                author,
                model_name,
                cv_strategy,
                metric_name,
                f"{cv_mean:.6f}",
                f"{cv_std:.6f}",
                seed,
                len(features),
                notes,
                "NO",
                "",
            ])

        print(f"[ExperimentLogger] Registrado {exp_id} | {metric_name}: {cv_mean:.6f} (+/- {cv_std:.6f})")
        return exp_id
