# Script de inicio rápido en local (Sin Docker) — Hackathon IA PUCP 2026
# Entrena/verifica el modelo activo y levanta el Backend FastAPI en el puerto 8000.

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

if (-not (Test-Path "product_app/backend/models/active_model.joblib")) {
    Write-Host "[1/2] Generando artefacto de modelo inicial reproducible..." -ForegroundColor Cyan
    python kaggle_pipeline/src/train.py
}

Write-Host "[2/2] Levantando API del Producto en http://localhost:8000 (Docs: http://localhost:8000/docs)" -ForegroundColor Green
Set-Location "product_app/backend"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
