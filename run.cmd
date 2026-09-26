@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
"%~dp0venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
