# Script de inicio rápido en local (Sin Docker) — INTI-3K · Hackathon IA PUCP 2026
# Verifica pesos oficiales (`models/modelo.pt`) y levanta el Backend FastAPI en el puerto 8000.

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

if (-not (Test-Path "models/modelo.pt")) {
    Write-Error "No se encontró models/modelo.pt. Ejecuta primero train/submission_notebook.ipynb o ./kaggle_pipeline/reproduce.ps1."
}
if (-not (Test-Path "data/static.npz")) {
    Write-Error "No se encontró data/static.npz. Descarga la data oficial en data/ siguiendo el README."
}

Write-Host "[INTI-3K] Levantando Backend FastAPI en http://localhost:8000 (Docs: http://localhost:8000/docs)" -ForegroundColor Green
Write-Host "[INTI-3K] En otra terminal ejecuta: cd product_app/frontend && npm install && npm run dev" -ForegroundColor Cyan

Set-Location "product_app/backend"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
