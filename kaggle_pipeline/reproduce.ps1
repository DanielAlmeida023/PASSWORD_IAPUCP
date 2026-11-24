# Script de Reproducibilidad en 1 Clic (Regla 7.e — Hackathon IA PUCP 2026)
# Recrea desde cero los folds, el entrenamiento con SEED=42 y las predicciones de Kaggle.

$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Reproduciendo Pipeline Oficial — Hackathon IA PUCP 2026  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

python src/train.py

Write-Host "✅ Reproducción completada con éxito (Seed = 42)." -ForegroundColor Green
