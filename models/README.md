# models/

Pesos del modelo final **`EnsembleV8`**: los mismos que generaron nuestra submission de Kaggle y los que carga el producto (`app/inti_inference.py`).

<!-- ACTUALIZAR CON MODELO FINAL: tamaño, SHA-256 y scores (deben coincidir con model_card.json) -->

| Archivo | Qué es |
|---|---|
| `modelo.pt` | `state_dict` de PyTorch (sin estado del optimizador) · **23.11 MB** (23,112,355 bytes) · SHA-256 `8a4ec8d1…e9a87e` |
| `norm_stats.npz` | Estadísticas de normalización del refit con el 100 % de la data oficial (0.87 MB) |
| `model_card.json` | Ficha: arquitectura, lags, pesos de mezcla por variable, skill de validación (0.4100, holdout 16–30 jun), LB público (0.49844) y SHA-256 de ambos archivos. El backend la sirve en `/api/model-info` |

- Se regeneran corriendo `train/submission_notebook.ipynb` con `TRAIN = True` (ver README principal).
- Pesan menos de 100 MB, así que van en el repositorio; no hay link externo.
- Para comprobar que tu copia es la correcta: `sha256sum models/modelo.pt` (o `Get-FileHash models\modelo.pt` en PowerShell) debe dar el valor de `model_card.json`.
