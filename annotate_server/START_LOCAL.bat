@echo off
setlocal
cd /d "%~dp0"
py -3 --version >nul 2>&1
if errorlevel 1 goto try_python
set "LOCAL_RAG_PY=py -3"
goto run
:try_python
python --version >nul 2>&1
if errorlevel 1 goto missing
set "LOCAL_RAG_PY=python"
:run
echo Open http://127.0.0.1:8765/ in your browser.
echo Keep this window open while annotating. Press Ctrl+C to stop.
%LOCAL_RAG_PY% -B annotate_server\annotation_server.py --host 127.0.0.1 --port 8765
pause
exit /b
:missing
echo Python 3.10 or newer is required. Install Python and run this file again.
pause
exit /b 1
