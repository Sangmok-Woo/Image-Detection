@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

rem Distilled student VLM server (torch venv, port 8502). Skipped if already running,
rem or if a remote server (Colab / AWS) is given by VLM_URL or a one-line vlm_url.txt.
if defined VLM_URL goto app
if exist "%~dp0vlm_url.txt" goto app
curl -s -o nul http://127.0.0.1:8502/health || (
  if exist "%~dp0venv-train\Scripts\python.exe" (
    start "Qwen2-VL server" /min "%~dp0venv-train\Scripts\python.exe" "%~dp0distill\vlm_server.py"
  ) else (
    echo [run] venv-train not found - VLM server skipped, MobileViT only.
  )
)

:app
"%~dp0venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
