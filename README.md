# Hackathon IA PUCP 2026 — Repositorio Oficial del Equipo

Solución integral para la **Hackathon IA PUCP 2026** (DSC PUCP × Grupo IA-PUCP), estructurada para cumplir al 100% con los dos componentes obligatorios de las bases:
1. **Componente 1 (`kaggle_pipeline/`):** Pipeline de Machine Learning competitivo, 100% reproducible (`SEED = 42`), entrenado exclusivamente con la data oficial de la competencia en Kaggle y con validación cruzada local (*Out-Of-Fold*).
2. **Componente 2 (`product_app/`):** Producto funcional full-stack (**FastAPI + Next.js/React**) que consume **datos actuales** y entrega soporte a la toma de decisiones para un usuario concreto.

---

## Estructura del Monorepo

```text
IA_PUCP/
├── AGENTS.md                          # Reglas estrictas del agente IA (Cero envíos automáticos, reproducibilidad)
├── .agents/
│   ├── rules/hackathon-ia-pucp.md     # Reglas de cumplimiento de las bases
│   └── skills/kaggle-competitive-ml/  # Flujo de trabajo de validación cruzada y verificación de submissions
├── kaggle_pipeline/                   # COMPONENTE 1: Modelo y Reproducibilidad Kaggle
│   ├── data/raw/                      # Datasets oficiales de Kaggle (ignorados en Git)
│   ├── experiments/
│   │   └── experiments_log.csv        # Bitácora compartida del equipo (CV score vs Public LB)
│   ├── src/
│   │   ├── seed.py                    # Fijación determinística de semillas (seed_everything(42))
│   │   ├── cv_strategy.py             # Generación de Folds sin data leakage
│   │   ├── experiment_logger.py       # Registro automático de experimentos y predicciones OOF
│   │   ├── check_submission.py        # Validador estricto pre-envío contra sample_submission.csv
│   │   └── train.py                   # Entrenamiento CV y exportación de artefactos al Producto
│   └── reproduce.ps1                  # Script de 1 clic para reproducir el submission (Regla 7.e)
├── product_app/                       # COMPONENTE 2: Producto Funcional con Datos Actuales
│   ├── backend/                       # API REST en FastAPI + Modelo activo exportado
│   └── frontend/                      # Interfaz web en Next.js + React
├── docker-compose.yml                 # Ejecución local de punta a punta en 1 comando
├── start_local.ps1                    # Arranque rápido local sin Docker
└── requirements.txt                   # Dependencias de Python unificadas
```

---

## Ejecución Rápida del Producto en Local (Para el Jurado / Preselección)

### Opción A: Con Docker Compose (1 solo comando)
```bash
docker compose up --build
```
- **Frontend (Producto Web):** `http://localhost:3000`
- **Backend API & Swagger Docs:** `http://localhost:8000/docs`

### Opción B: Ejecución directa en Windows / Linux / macOS
1. Instalar dependencias de Python y entrenar/exportar el modelo activo:
   ```bash
   pip install -r requirements.txt
   python kaggle_pipeline/src/train.py
   ```
2. Levantar el Backend (FastAPI):
   ```bash
   cd product_app/backend
   uvicorn app.main:app --reload --port 8000
   ```
3. En otra terminal, levantar el Frontend (Next.js):
   ```bash
   cd product_app/frontend
   npm install
   npm run dev
   ```

---

## Reproducibilidad del Resultado en Kaggle (Regla 7.e)

Para reproducir exactamente las predicciones enviadas a Kaggle a partir de la data oficial ubicada en `kaggle_pipeline/data/raw/`:

```powershell
./kaggle_pipeline/reproduce.ps1
```

Antes de subir cualquier archivo a Kaggle, validar su integridad con:
```bash
python kaggle_pipeline/src/check_submission.py --sub <ruta_submission.csv> --sample kaggle_pipeline/data/raw/sample_submission.csv
```
