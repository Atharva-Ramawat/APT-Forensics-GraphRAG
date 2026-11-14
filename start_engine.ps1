<#
.SYNOPSIS
    Autonomous APT Forensics Hybrid GraphRAG Engine - Pipeline Orchestrator
.DESCRIPTION
    Automates Phase 2 QLoRA fine-tuning, launches the Phase 3 FastAPI backend in
    a background PowerShell process, and launches the Phase 4 Streamlit UI.
#>

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [1/4] Installing Unsloth & Fine-Tuning Dependencies for Phase 2..." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
pip install unsloth trl peft accelerate bitsandbytes

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " [2/4] Executing Phase 2: Qwen2.5-Coder-3B QLoRA Extractor Training..." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
python train_extractor.py

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " [3/4] Phase 2 Execution Completed. Verify model artifacts above." -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Pause

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " [4/4] Launching Phase 3 FastAPI Backend in a new window..." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Launching Phase 4 Streamlit UI in the current window..." -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
python -m streamlit run streamlit_app.py
