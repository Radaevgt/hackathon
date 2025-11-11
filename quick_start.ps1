# Quick Start Script для Alpha RAG System
# Запуск полного pipeline от начала до конца

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  ALPHA RAG SYSTEM - QUICK START" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

# Проверка виртуального окружения
if (-not $env:VIRTUAL_ENV) {
    Write-Host "`n⚠️  Virtual environment not activated" -ForegroundColor Yellow
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✅ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green

# Шаг 1: Проверка окружения
Write-Host "`n[1/3] Checking environment..." -ForegroundColor Cyan
python setup_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Environment check failed" -ForegroundColor Red
    exit 1
}

# Шаг 2: Построение индексов
Write-Host "`n[2/3] Building indices..." -ForegroundColor Cyan
python scripts/build_indices.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Index building failed" -ForegroundColor Red
    exit 1
}

# Шаг 3: Запуск retrieval
Write-Host "`n[3/3] Running retrieval..." -ForegroundColor Cyan
python scripts/run_retrieval.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Retrieval failed" -ForegroundColor Red
    exit 1
}

# Финал
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "✅ PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "`nSubmission file: data/submissions/submission.csv" -ForegroundColor Yellow
Write-Host "Logs: logs/" -ForegroundColor Yellow
Write-Host "`nTo view results:" -ForegroundColor Cyan
Write-Host "  Get-Content data/submissions/submission.csv | Select-Object -First 10" -ForegroundColor White
