"""
Utilidad de reproducibilidad determinística para la Hackathon IA PUCP 2026 (Regla 7.e).
Fija semillas en Python, NumPy, variables de entorno y PyTorch (si está instalado).
"""

import os
import random
import numpy as np

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Fija todas las semillas aleatorias para garantizar 100% de reproducibilidad."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
