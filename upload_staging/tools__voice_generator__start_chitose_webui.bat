@echo off
chcp 65001 >nul
title Chitose Voice Lab

rem Edit these three paths before the first run.
set "GPT_SOVITS_ROOT=C:\path\to\GPT-SoVITS"
set "GPT_SOVITS_PYTHON=C:\path\to\GPT-SoVITS\runtime\python.exe"
set "CHITOSE_CONFIG_PATH=C:\path\to\chitose_v0.1.0_inference.yaml"

set "CHITOSE_OUTPUT_DIR=%~dp0outputs"
set "PYTHONPATH=%GPT_SOVITS_ROOT%;%GPT_SOVITS_ROOT%\GPT_SoVITS"
set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUTF8=1"

if not exist "%GPT_SOVITS_PYTHON%" (
  echo [ERROR] Python runtime not found: %GPT_SOVITS_PYTHON%
  pause
  exit /b 1
)
if not exist "%CHITOSE_CONFIG_PATH%" (
  echo [ERROR] Inference config not found: %CHITOSE_CONFIG_PATH%
  pause
  exit /b 1
)

cd /d "%GPT_SOVITS_ROOT%"
"%GPT_SOVITS_PYTHON%" -s "%~dp0chitose_webui.py"

echo.
echo Chitose Voice Lab has stopped.
pause

